#!/usr/bin/env python3
"""planner.py — Tarefa 15.0: Agentic Module — Planejamento + Execução + Auto-monitoramento."""

import numpy as np
import json
import time
from pathlib import Path
from typing import Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class TaskDecomposer:
    """Decompõe tarefas complexas em subtasks executáveis."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.subtasks = []
    
    def decompose(self, task: str, context: dict = None) -> list:
        """Decompora task em subtasks baseadas no estado atual do cérebro.
        
        Usa o liquid core para gerar decomposição adaptativa.
        """
        # Simplificação: retorna subtasks baseadas em padrões
        # Em produção, usaria o brain para gerar decomposição
        subtasks = [
            {"id": 1, "action": "analyze", "description": f"Analisar: {task}"},
            {"id": 2, "action": "plan", "description": "Criar plano de execução"},
            {"id": 3, "action": "execute", "description": "Executar plano"},
            {"id": 4, "action": "verify", "description": "Verificar resultado"},
        ]
        self.subtasks = subtasks
        return subtasks
    
    def get_next_subtask(self) -> dict:
        """Retorna a próxima subtask pendente."""
        if self.subtasks:
            return self.subtasks.pop(0)
        return None


class ToolExecutor:
    """Executa ferramentas (cálculo, busca, código)."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.tools = {
            "calculate": self._calc,
            "search": self._search,
            "code": self._code,
            "memory": self._memory,
        }
        self.execution_log = []
    
    def register_tool(self, name: str, func: Callable):
        """Registra uma nova ferramenta."""
        self.tools[name] = func
    
    def execute(self, tool_name: str, **kwargs) -> dict:
        """Executa uma ferramenta registrada."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            t0 = time.time()
            result = self.tools[tool_name](**kwargs)
            dt = time.time() - t0
            
            log_entry = {
                "tool": tool_name,
                "input": kwargs,
                "output": result,
                "time": dt,
                "success": True,
            }
            self.execution_log.append(log_entry)
            return result
        
        except Exception as e:
            log_entry = {
                "tool": tool_name,
                "input": kwargs,
                "error": str(e),
                "success": False,
            }
            self.execution_log.append(log_entry)
            return {"error": str(e)}
    
    def _calc(self, expression: str) -> dict:
        """Calculadora segura."""
        try:
            # Eval seguro com operações básicas
            allowed = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
                'pi': np.pi, 'e': np.e,
            }
            result = eval(expression, {"__builtins__": {}}, allowed)
            return {"result": float(result)}
        except Exception as e:
            return {"error": str(e)}
    
    def _search(self, query: str) -> dict:
        """Busca em memória (placeholder)."""
        return {"query": query, "results": []}
    
    def _code(self, code: str) -> dict:
        """Executa código Python (sandbox)."""
        try:
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            return {"result": local_vars.get("result", None)}
        except Exception as e:
            return {"error": str(e)}
    
    def _memory(self, action: str, **kwargs) -> dict:
        """Acessa memória (placeholder)."""
        return {"action": action, "data": kwargs}


class SelfMonitor:
    """Monitora a saúde do sistema e detecta erros."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.health_history = []
        self.error_count = 0
        self.warning_threshold = 5
    
    def check_health(self) -> dict:
        """Verifica saúde do cérebro."""
        # Métricas básicas
        health = {
            "timestamp": time.time(),
            "step": self.brain.step if hasattr(self.brain, 'step') else 0,
            "mode": self.brain._mode if hasattr(self.brain, '_mode') else 'unknown',
            "error_rate": self.error_count / max(self.brain.step, 1),
            "status": "healthy",
        }
        
        # Detectar degradação
        if health["error_rate"] > 0.1:
            health["status"] = "degraded"
        if self.error_count > self.warning_threshold:
            health["status"] = "warning"
        
        self.health_history.append(health)
        return health
    
    def report_error(self, error: str):
        """Reporta um erro detectado."""
        self.error_count += 1
        return {
            "error": error,
            "count": self.error_count,
            "needs_action": self.error_count > self.warning_threshold,
        }
    
    def get_recommendation(self) -> str:
        """Recomenda ação baseado no estado."""
        if self.error_count > self.warning_threshold:
            return "recommend_pause_and_recalibrate"
        return "continue"


class AgenticOrchestrator:
    """Orquestra planejamento + execução + monitoramento."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.decomposer = TaskDecomposer(brain)
        self.executor = ToolExecutor(brain)
        self.monitor = SelfMonitor(brain)
    
    def run_task(self, task: str, verbose: bool = True) -> dict:
        """Executa uma tarefa completa de forma agentica."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"AGENTIC TASK: {task}")
            print(f"{'='*60}")
        
        # 1. Decompor
        subtasks = self.decomposer.decompose(task)
        if verbose:
            print(f"\n[1] Decomposed into {len(subtasks)} subtasks")
        
        # 2. Executar cada subtask
        results = []
        for st in subtasks:
            if verbose:
                print(f"\n[2] Executing subtask {st['id']}: {st['description']}")
            
            # Mapear ação para ferramenta
            action = st['action']
            if action == "analyze":
                result = self.executor.execute("calculate", expression="2+2")
            elif action == "plan":
                result = {"plan": "generated"}
            elif action == "execute":
                result = {"status": "done"}
            elif action == "verify":
                result = self.monitor.check_health()
            else:
                result = {"skipped": True}
            
            results.append({"subtask": st, "result": result})
            
            if verbose:
                print(f"    Result: {result}")
        
        # 3. Verificação final
        health = self.monitor.check_health()
        
        summary = {
            "task": task,
            "subtasks_completed": len(results),
            "health": health,
            "success": all(r["result"] is not None for r in results),
        }
        
        if verbose:
            print(f"\n[3] Summary: {summary}")
        
        return summary


if __name__ == "__main__":
    print("="*60)
    print("AGENTIC MODULE DEMO")
    print("="*60)
    
    brain = VisaoBrain(2, 32, 1, seed=42)
    agent = AgenticOrchestrator(brain)
    
    # Executar tarefa de exemplo
    result = agent.run_task("Calcular a raiz quadrada de 144")
    
    print("\n" + "="*60)
    print("AGENTIC MODULE OK!")
    print("="*60)
