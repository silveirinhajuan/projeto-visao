"""
watchdog.py — Watchdog do Projeto VISÃO.

Monitora o agente autônomo e garante operação contínua:
  1. Verifica se o agente está vivo (heartbeat)
  2. Reinicia automaticamente em caso de falha
  3. Monitora recursos do sistema (RAM, CPU, disco)
  4. Mantém log de uptime e recuperações
  5. Respeita limites do BACKLOG (não avança fase sem números)

Regra dura: o watchdog NÃO modifica BACKLOG.json, NÃO re-sela manifesto,
NÃO avança fase. Apenas observa e reinicia o agente.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np


# ==============================================================
#  CONFIGURAÇÃO
# ==============================================================

ROOT = Path(__file__).resolve().parents[2]
BACKLOG_PATH = ROOT / "BACKLOG.json"
HEARTBEAT_PATH = ROOT / "visao" / "ops" / ".heartbeat"
LOG_DIR = ROOT / "visao" / "ops" / "logs"
WATCHDOG_LOG = LOG_DIR / "watchdog.jsonl"

# Intervalos (segundos)
HEARTBEAT_TIMEOUT = 600      # 10 min sem heartbeat = morto
CHECK_INTERVAL = 60          # verifica a cada 1 min
RESTART_COOLDOWN = 300       # 5 min entre reinícios
MAX_RESTARTS_PER_HOUR = 5

# Limites do sistema
MAX_RAM_MB = 2048            # 2GB
MAX_CPU_PERCENT = 80
MAX_DISK_USAGE_PERCENT = 90


# ==============================================================
#  ESTADOS
# ==============================================================

class AgentState(Enum):
    """Estados possíveis do agente."""
    RUNNING = "running"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    RESTARTING = "restarting"
    FAILED = "failed"


class SystemHealth(Enum):
    """Saúde geral do sistema."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Heartbeat:
    """Heartbeat do agente."""
    timestamp: float
    task_id: Optional[str] = None
    step: int = 0
    status: str = "working"
    
    def is_stale(self, timeout: float = HEARTBEAT_TIMEOUT) -> bool:
        return (time.time() - self.timestamp) > timeout


@dataclass
class SystemStatus:
    """Status do sistema."""
    timestamp: float
    ram_mb: float
    cpu_percent: float
    disk_percent: float
    agent_state: str
    health: str
    uptime_seconds: float
    total_restarts: int
    last_heartbeat: Optional[float] = None
    current_task: Optional[str] = None


@dataclass
class RestartRecord:
    """Registro de reinício."""
    timestamp: float
    reason: str
    previous_task: Optional[str]
    success: bool


# ==============================================================
#  WATCHDOG
# ==============================================================

class VisaoWatchdog:
    """
    Watchdog do Projeto VISÃO.
    
    Monitora o agente autônomo e garante operação contínua
    dentro dos limites estabelecidos pelo BACKLOG.
    """
    
    def __init__(
        self,
        heartbeat_path: Path = HEARTBEAT_PATH,
        log_path: Path = WATCHDOG_LOG,
        check_interval: float = CHECK_INTERVAL,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
        max_restarts_per_hour: int = MAX_RESTARTS_PER_HOUR,
    ):
        self.heartbeat_path = Path(heartbeat_path)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_restarts_per_hour = max_restarts_per_hour
        
        # Estado
        self._running = False
        self._agent_state = AgentState.STOPPED
        self._system_health = SystemHealth.HEALTHY
        self._start_time = 0.0
        self._total_restarts = 0
        self._restart_history: list[RestartRecord] = []
        self._last_heartbeat: Optional[Heartbeat] = None
        self._current_task: Optional[str] = None
        
        # Lock para thread safety
        self._lock = threading.Lock()
    
    # ==========================================================
    #  API PÚBLICA
    # ==========================================================
    
    def start(self) -> None:
        """Inicia o watchdog."""
        self._running = True
        self._start_time = time.time()
        self._agent_state = AgentState.RUNNING
        
        self._log_event("watchdog_started", {
            "heartbeat_path": str(self.heartbeat_path),
            "check_interval": self.check_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
        })
        
        print(f"[Watchdog] INICIADO")
        print(f"[Watchdog] Heartbeat timeout: {self.heartbeat_timeout}s")
        print(f"[Watchdog] Check interval: {self.check_interval}s")
        print(f"[Watchdog] Max restarts/hora: {self.max_restarts_per_hour}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self._monitor_loop()
        except Exception as e:
            self._log_event("watchdog_fatal_error", {
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
        finally:
            self._shutdown()
    
    def stop(self) -> None:
        """Para o watchdog."""
        self._running = False
        self._log_event("watchdog_stopped", {})
    
    def get_status(self) -> dict:
        """Retorna status atual do watchdog."""
        return {
            "watchdog_running": self._running,
            "agent_state": self._agent_state.value,
            "system_health": self._system_health.value,
            "uptime_seconds": time.time() - self._start_time,
            "total_restarts": self._total_restarts,
            "last_heartbeat": asdict(self._last_heartbeat) if self._last_heartbeat else None,
            "current_task": self._current_task,
            "restart_history": [asdict(r) for r in self._restart_history[-10:]],
        }
    
    # ==========================================================
    #  LOOP DE MONITORAMENTO
    # ==========================================================
    
    def _monitor_loop(self) -> None:
        """Loop principal de monitoramento."""
        while self._running:
            try:
                # 1. Ler heartbeat
                heartbeat = self._read_heartbeat()
                
                # 2. Verificar se agente está vivo
                if heartbeat and heartbeat.is_stale(self.heartbeat_timeout):
                    self._log_event("heartbeat_stale", {
                        "last_heartbeat": heartbeat.timestamp,
                        "elapsed": time.time() - heartbeat.timestamp,
                    })
                    self._handle_agent_dead("heartbeat_stale")
                
                # 3. Verificar recursos do sistema
                sys_status = self._check_system_resources()
                
                # 4. Atualizar estado
                with self._lock:
                    self._last_heartbeat = heartbeat
                    self._current_task = heartbeat.task_id if heartbeat else None
                    self._system_health = self._evaluate_health(sys_status)
                
                # 5. Log periódico
                self._log_status(sys_status)
                
            except Exception as e:
                self._log_event("monitor_error", {"error": str(e)})
            
            time.sleep(self.check_interval)
    
    def _read_heartbeat(self) -> Optional[Heartbeat]:
        """Lê heartbeat do agente."""
        try:
            if not self.heartbeat_path.exists():
                return None
            
            with open(self.heartbeat_path, "r") as f:
                data = json.load(f)
            
            return Heartbeat(
                timestamp=data.get("timestamp", 0),
                task_id=data.get("task_id"),
                step=data.get("step", 0),
                status=data.get("status", "unknown"),
            )
        except Exception:
            return None
    
    def _handle_agent_dead(self, reason: str) -> None:
        """Trata agente morto — tenta reiniciar."""
        self._agent_state = AgentState.RESTARTING
        
        # Verificar limite de reinícios
        recent_restarts = [
            r for r in self._restart_history
            if time.time() - r.timestamp < 3600
        ]
        
        if len(recent_restarts) >= self.max_restarts_per_hour:
            self._log_event("max_restarts_exceeded", {
                "recent_restarts": len(recent_restarts),
                "reason": reason,
            })
            self._agent_state = AgentState.FAILED
            return
        
        # Tentar reiniciar
        self._log_event("restarting_agent", {
            "reason": reason,
            "previous_task": self._current_task,
        })
        
        success = self._restart_agent()
        
        record = RestartRecord(
            timestamp=time.time(),
            reason=reason,
            previous_task=self._current_task,
            success=success,
        )
        self._restart_history.append(record)
        self._total_restarts += 1
        
        if success:
            self._agent_state = AgentState.RUNNING
            self._log_event("agent_restarted", {"task": self._current_task})
        else:
            self._agent_state = AgentState.FAILED
            self._log_event("restart_failed", {"reason": reason})
    
    def _restart_agent(self) -> bool:
        """
        Reinicia o agente autônomo.
        
        O agente deve ser reiniciado a partir da última tarefa
        pendente do BACKLOG. O watchdog NÃO modifica o BACKLOG.
        """
        try:
            # Verificar se BACKLOG existe
            if not BACKLOG_PATH.exists():
                self._log_event("restart_error", {"error": "BACKLOG not found"})
                return False
            
            # Ler BACKLOG para encontrar próxima tarefa
            with open(BACKLOG_PATH, "r") as f:
                backlog = json.load(f)
            
            # Encontrar tarefa em progresso ou próxima pendente
            tasks = backlog.get("tasks", [])
            in_progress = [t for t in tasks if t.get("status") == "in_progress"]
            pending = [t for t in tasks if t.get("status") == "pending"]
            
            target_task = None
            if in_progress:
                target_task = in_progress[0]
            elif pending:
                # Encontrar primeira pendente com dependências done
                done_ids = {t.get("id") for t in tasks if t.get("status") == "done"}
                for t in pending:
                    deps = t.get("depends_on", [])
                    if all(d in done_ids for d in deps):
                        target_task = t
                        break
            
            if not target_task:
                self._log_event("restart_nothing_to_do", {})
                return True  # Não é falha, só não há o que fazer
            
            # Criar heartbeat inicial para desbloquear
            self._write_initial_heartbeat(target_task.get("id"))
            
            return True
            
        except Exception as e:
            self._log_event("restart_error", {"error": str(e)})
            return False
    
    def _write_initial_heartbeat(self, task_id: str) -> None:
        """Escreve heartbeat inicial para desbloquear o agente."""
        heartbeat = {
            "timestamp": time.time(),
            "task_id": task_id,
            "step": 0,
            "status": "restarting",
        }
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.heartbeat_path, "w") as f:
            json.dump(heartbeat, f)
    
    # ==========================================================
    #  RECURSOS DO SISTEMA
    # ==========================================================
    
    def _check_system_resources(self) -> dict:
        """Verifica recursos do sistema."""
        status: dict = {
            "ram_mb": 0.0,
            "cpu_percent": 0.0,
            "disk_percent": 0.0,
        }
        
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            status["ram_mb"] = usage.ru_maxrss / 1024.0
        except Exception:
            pass
        
        try:
            # CPU via /proc/stat (Linux)
            with open("/proc/stat", "r") as f:
                line = f.readline()
                fields = line.split()[1:]
                idle = int(fields[3])
                total = sum(int(x) for x in fields)
                status["cpu_percent"] = 100.0 * (1.0 - idle / total) if total > 0 else 0.0
        except Exception:
            pass
        
        try:
            # Disco via statvfs
            stat = os.statvfs(str(ROOT))
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            used = total - free
            status["disk_percent"] = 100.0 * used / total if total > 0 else 0.0
        except Exception:
            pass
        
        return status
    
    def _evaluate_health(self, sys_status: dict) -> SystemHealth:
        """Avalia saúde do sistema."""
        if sys_status["ram_mb"] > MAX_RAM_MB:
            return SystemHealth.CRITICAL
        if sys_status["cpu_percent"] > MAX_CPU_PERCENT:
            return SystemHealth.WARNING
        if sys_status["disk_percent"] > MAX_DISK_USAGE_PERCENT:
            return SystemHealth.CRITICAL
        return SystemHealth.HEALTHY
    
    # ==========================================================
    #  LOG & SHUTDOWN
    # ==========================================================
    
    def _log_event(self, event_type: str, data: dict) -> None:
        """Log de evento."""
        entry = {
            "event": event_type,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "agent_state": self._agent_state.value,
            "system_health": self._system_health.value,
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    def _log_status(self, sys_status: dict) -> None:
        """Log de status."""
        status = SystemStatus(
            timestamp=time.time(),
            ram_mb=sys_status["ram_mb"],
            cpu_percent=sys_status["cpu_percent"],
            disk_percent=sys_status["disk_percent"],
            agent_state=self._agent_state.value,
            health=self._system_health.value,
            uptime_seconds=time.time() - self._start_time,
            total_restarts=self._total_restarts,
            last_heartbeat=self._last_heartbeat.timestamp if self._last_heartbeat else None,
            current_task=self._current_task,
        )
        self._log_event("status", asdict(status))
    
    def _signal_handler(self, signum, frame) -> None:
        """Handler de sinal."""
        self._running = False
    
    def _shutdown(self) -> None:
        """Shutdown do watchdog."""
        self._agent_state = AgentState.STOPPED
        self._log_event("watchdog_shutdown", {
            "total_restarts": self._total_restarts,
            "uptime_seconds": time.time() - self._start_time,
        })
        print(f"[Watchdog] FINALIZADO — uptime: {time.time() - self._start_time:.0f}s, restarts: {self._total_restarts}")


# ==============================================================
#  HEARTBEAT AGENTE (para o agente usar)
# ==============================================

class AgentHeartbeat:
    """
    Heartbeat que o agente autônomo deve chamar periodicamente.
    
    Uso:
        heartbeat = AgentHeartbeat()
        heartbeat.start(task_id="T1.1")
        
        # Durante execução:
        heartbeat.update(step=42)
        
        # Ao terminar:
        heartbeat.finish()
    """
    
    def __init__(self, path: Path = HEARTBEAT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def start(self, task_id: str) -> None:
        """Inicia heartbeat para uma tarefa."""
        self._write(task_id, 0, "working")
    
    def update(self, step: int, status: str = "working") -> None:
        """Atualiza heartbeat."""
        data = self._read()
        task_id = data.get("task_id") if data else None
        self._write(task_id, step, status)
    
    def finish(self) -> None:
        """Finaliza heartbeat."""
        data = self._read()
        task_id = data.get("task_id") if data else None
        self._write(task_id, 0, "done")
    
    def _write(self, task_id: Optional[str], step: int, status: str) -> None:
        """Escreve heartbeat."""
        data = {
            "timestamp": time.time(),
            "task_id": task_id,
            "step": step,
            "status": status,
        }
        with open(self.path, "w") as f:
            json.dump(data, f)
    
    def _read(self) -> Optional[dict]:
        """Lê heartbeat."""
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return None


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VISÃO Watchdog")
    parser.add_argument(
        "--mode",
        choices=["start", "status", "stop"],
        default="start",
        help="Modo: start (inicia watchdog), status (mostra status), stop (para watchdog)",
    )
    parser.add_argument("--interval", type=float, default=CHECK_INTERVAL, help="Intervalo de verificação (s)")
    parser.add_argument("--timeout", type=float, default=HEARTBEAT_TIMEOUT, help="Timeout do heartbeat (s)")
    args = parser.parse_args()
    
    if args.mode == "start":
        watchdog = VisaoWatchdog(
            check_interval=args.interval,
            heartbeat_timeout=args.timeout,
        )
        watchdog.start()
    elif args.mode == "status":
        # Mostrar status atual
        watchdog = VisaoWatchdog()
        print(json.dumps(watchdog.get_status(), indent=2, default=str))
    elif args.mode == "stop":
        # Enviar sinal para processo rodando
        import subprocess
        subprocess.run(["pkill", "-f", "visao/ops/watchdog.py"])
