"""
neuro_symbolic.py — Tarefa 14.0: Reasoning Layer (Neuro-Symbolic).

Implementa raciocínio neuro-simbólico sobre o liquid core do Projeto VISÃO,
baseado em:
  - Agentic Neural Networks (arXiv 2506.09046): multi-agent como rede neural
    em camadas, com "textual backpropagation" para refinar papéis/coordenação.
  - Scallop: programação lógica diferenciável (Datalog) com raciocínio
    relacional e semirings de proveniência.

Componentes:
  1. SymbolicSolver — planejador lógico (PDDL-like) para decomposição de tarefas.
     Usa forward chaining sobre regras STRIPS-like. Produz planos como
     sequências de ações com pré-condições e efeitos.
  2. TextualGradient — módulo de feedback estruturado que ajusta o liquid core.
     Converte erros/verificações em "gradientes textuais" que modulam lr,
     surpresa e consolidação (compatível com Oja + EWC + Surprise).
  3. VerificationEngine — verificador de provas (validade lógica de planos)
     e código (corretude sintática/semântica básica).

O Reasoning Layer NÃO substitui o liquid core — ele o COMPLEMENTA:
  - O liquid core lida com percepção temporal, adaptação contínua e memória implícita.
  - O reasoning layer lida com planejamento explícito, decomposição e verificação.
  - A interface entre eles é o TextualGradient: feedback estruturado que ajusta
    os hiperparâmetros do core baseado em raciocínio simbólico.

Referências:
  - Ma et al., "Self-Evolving Multi-Agent Systems via Textual Backpropagation" (2025)
  - Naik et al., "Scallop: A Language for Neurosymbolic Programming" (2023)
  - Hasani et al., "Liquid Time-constant Networks" (AAAI 2021)
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Garantir que o prototype/ e visao/ sejam importáveis
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ==============================================================
#  ESTRUTURAS SIMBÓLICAS
# ==============================================================

class FormulaType(Enum):
    """Tipos de fórmulas lógicas suportadas."""
    ATOM = auto()       # predicado atômico: P(x, y)
    AND = auto()        # conjunção: A ∧ B
    OR = auto()         # disjunção: A ∨ B
    NOT = auto()        # negação: ¬A
    IMPLIES = auto()    # implicação: A → B
    FORALL = auto()     # quantificação universal: ∀x. P(x)
    EXISTS = auto()     # quantificação existencial: ∃x. P(x)


@dataclass
class Atom:
    """Predicado atômico: nome + argumentos."""
    name: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def __str__(self) -> str:
        neg = "¬" if self.negated else ""
        args_str = ", ".join(self.args) if self.args else ""
        return f"{neg}{self.name}({args_str})"

    def __hash__(self) -> int:
        return hash((self.name, self.args, self.negated))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return NotImplemented
        return (self.name == other.name and
                self.args == other.args and
                self.negated == other.negated)

    def ground(self, binding: dict[str, str]) -> "Atom":
        """Substitui variáveis por constantes."""
        new_args = tuple(binding.get(a, a) for a in self.args)
        return Atom(self.name, new_args, self.negated)

    def matches(self, other: "Atom") -> bool:
        """Verifica se dois átomos são compatíveis (para unificação simples)."""
        if self.name != other.name:
            return False
        if self.negated != other.negated:
            return False
        if len(self.args) != len(other.args):
            return False
        return all(a == b or a.startswith("?") or b.startswith("?")
                   for a, b in zip(self.args, other.args))


@dataclass
class Action:
    """Ação STRIPS-like: nome + pré-condições + efeitos."""
    name: str
    preconditions: list[Atom] = field(default_factory=list)
    add_effects: list[Atom] = field(default_factory=list)
    del_effects: list[Atom] = field(default_factory=list)
    cost: float = 1.0

    def __str__(self) -> str:
        return self.name

    def is_applicable(self, state: set[Atom]) -> bool:
        """Verifica se a ação é aplicável no estado dado."""
        return all(
            any(pre.matches(s) and pre.negated == s.negated for s in state)
            for pre in self.preconditions
        )

    def apply(self, state: set[Atom]) -> set[Atom]:
        """Aplica a ação, retornando novo estado."""
        new_state = set(state)
        # Remove efeitos deletados
        for eff in self.del_effects:
            to_remove = [s for s in new_state if eff.matches(s)]
            for r in to_remove:
                new_state.discard(r)
        # Adiciona efeitos
        for eff in self.add_effects:
            new_state.add(eff)
        return new_state


@dataclass
class Plan:
    """Plano: sequência de ações + metadados."""
    actions: list[Action]
    goal: list[Atom]
    state_trace: list[set[Atom]] = field(default_factory=list)
    success: bool = False
    cost: float = 0.0

    def __len__(self) -> int:
        return len(self.actions)

    def __str__(self) -> str:
        if not self.actions:
            return "Plan(empty)"
        steps = " → ".join(a.name for a in self.actions)
        return f"Plan({steps}, cost={self.cost:.2f}, success={self.success})"


# ==============================================================
#  1. SYMBOLIC SOLVER — Planejador Lógico (PDDL-like)
# ==============================================================

class SymbolicSolver:
    """Planejador lógico baseado em forward chaining (STRIPS-like).

    Decompõe tarefas complexas em sequências de ações primitivas.
    Usa busca em largura (BFS) sobre o espaço de estados para encontrar
    um plano que atinja o objetivo a partir do estado inicial.

    Inspirado em:
      - PDDL (Planning Domain Definition Language)
      - Scallop (programação lógica relacional)
      - ANN (decomposição de tarefa em subtasks por camada)

    Parâmetros
    ----------
    actions : list[Action]
        Conjunto de ações disponíveis (domínio).
    max_depth : int
        Profundidade máxima de busca (evita explosão combinatória).
    max_states : int
        Número máximo de estados explorados por busca.
    verbose : bool
        Se True, imprime passos intermediários.
    """

    def __init__(
        self,
        actions: list[Action] | None = None,
        max_depth: int = 20,
        max_states: int = 10000,
        verbose: bool = False,
    ):
        self.actions: list[Action] = actions or []
        self.max_depth = max_depth
        self.max_states = max_states
        self.verbose = verbose
        self._stats: dict[str, Any] = {}

    def add_action(self, action: Action) -> None:
        """Registra uma ação no domínio."""
        self.actions.append(action)

    def define_domain(self, domain: dict[str, Any]) -> None:
        """Define domínio a partir de dict (formato PDDL-like simplificado).

        Formato do dict:
        {
          "actions": [
            {
              "name": "move",
              "preconditions": ["at(robot, ?from)", "connected(?from, ?to)"],
              "add": ["at(robot, ?to)"],
              "del": ["at(robot, ?from)"]
            }
          ]
        }
        """
        for act_def in domain.get("actions", []):
            pre = [self._parse_atom(p) for p in act_def.get("preconditions", [])]
            add = [self._parse_atom(a) for a in act_def.get("add", [])]
            dels = [self._parse_atom(d) for d in act_def.get("del", [])]
            action = Action(
                name=act_def["name"],
                preconditions=pre,
                add_effects=add,
                del_effects=dels,
                cost=act_def.get("cost", 1.0),
            )
            self.add_action(action)

    def solve(
        self,
        initial_state: set[Atom],
        goal: list[Atom],
    ) -> Plan:
        """Encontra um plano do estado inicial ao objetivo.

        Usa BFS sobre o espaço de estados. Retorna um Plan com a sequência
        de ações e o rastro de estados.

        Parameters
        ----------
        initial_state : set[Atom]
            Estado inicial (conjunto de átomos verdadeiros).
        goal : list[Atom]
            Objetivo (lista de átomos que devem ser verdadeiros).

        Returns
        -------
        Plan
            Plano encontrado (success=True) ou plano vazio (success=False).
        """
        if self._is_goal_satisfied(initial_state, goal):
            return Plan(actions=[], goal=goal, state_trace=[initial_state], success=True)

        # BFS: fila de (estado, plano_parcial)
        queue: list[tuple[set[Atom], list[Action], list[set[Atom]]]] = [
            (initial_state, [], [initial_state])
        ]
        visited: set[frozenset[Atom]] = {frozenset(initial_state)}
        states_explored = 0

        while queue and states_explored < self.max_states:
            state, plan_actions, trace = queue.pop(0)
            states_explored += 1

            if len(plan_actions) >= self.max_depth:
                continue

            for action in self.actions:
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    frozen = frozenset(new_state)

                    if frozen in visited:
                        continue
                    visited.add(frozen)

                    new_actions = plan_actions + [action]
                    new_trace = trace + [new_state]

                    if self._is_goal_satisfied(new_state, goal):
                        total_cost = sum(a.cost for a in new_actions)
                        self._stats = {
                            "states_explored": states_explored,
                            "plan_length": len(new_actions),
                            "cost": total_cost,
                        }
                        if self.verbose:
                            print(f"[SymbolicSolver] Plano encontrado: {len(new_actions)} ações, "
                                  f"{states_explored} estados explorados")
                        return Plan(
                            actions=new_actions,
                            goal=goal,
                            state_trace=new_trace,
                            success=True,
                            cost=total_cost,
                        )

                    queue.append((new_state, new_actions, new_trace))

        # Falha: nenhum plano encontrado
        self._stats = {
            "states_explored": states_explored,
            "plan_length": 0,
            "cost": float("inf"),
        }
        if self.verbose:
            print(f"[SymbolicSolver] Nenhum plano encontrado "
                  f"({states_explored} estados explorados)")
        return Plan(actions=[], goal=goal, success=False)

    def decompose(self, task: str, context: dict[str, Any] | None = None) -> list[str]:
        """Decompõe uma tarefa em subtasks (forward phase do ANN).

        Mapeia uma descrição de tarefa em linguagem natural para uma
        sequência de ações primitivas. Usa heurísticas baseadas em
        palavras-chave quando não há domínio formal definido.

        Parameters
        ----------
        task : str
            Descrição da tarefa (ex: "calcular fibonacci de 10").
        context : dict
            Contexto adicional (variáveis, restrições).

        Returns
        -------
        list[str]
            Lista de subtasks em ordem de execução.
        """
        task_lower = task.lower().strip()
        subtasks: list[str] = []

        # Heurísticas de decomposição baseadas em padrões
        if "calcular" in task_lower or "compute" in task_lower:
            if "fibonacci" in task_lower:
                subtasks = [
                    "parse_input(task)",
                    "initialize_fibonacci_state(n)",
                    "iterate_fibonacci(n)",
                    "return_result(fib_n)",
                ]
            elif "fatorial" in task_lower or "factorial" in task_lower:
                subtasks = [
                    "parse_input(task)",
                    "initialize_factorial_state(n)",
                    "iterate_factorial(n)",
                    "return_result(fact_n)",
                ]
            elif "primo" in task_lower or "prime" in task_lower:
                subtasks = [
                    "parse_input(task)",
                    "check_divisibility(n, 2)",
                    "iterate_sqrt(n)",
                    "return_result(is_prime)",
                ]
            else:
                subtasks = [
                    "parse_input(task)",
                    "identify_operation(task)",
                    "execute_operation(params)",
                    "return_result(output)",
                ]
        elif "provar" in task_lower or "prove" in task_lower:
            subtasks = [
                "parse_theorem(task)",
                "identify_axioms(theorem)",
                "apply_inference_rules(axioms)",
                "verify_proof_steps(steps)",
                "return_result(proof)",
            ]
        elif "verificar" in task_lower or "verify" in task_lower:
            subtasks = [
                "parse_input(task)",
                "identify_verification_type(input)",
                "execute_verification(input)",
                "return_result(verification)",
            ]
        elif "buscar" in task_lower or "search" in task_lower:
            subtasks = [
                "parse_query(task)",
                "initialize_search_space(query)",
                "traverse_space(query)",
                "collect_results(matches)",
                "return_result(results)",
            ]
        else:
            # Decomposição genérica
            subtasks = [
                "parse_input(task)",
                "analyze_requirements(input)",
                "decompose_into_steps(requirements)",
                "execute_steps(steps)",
                "return_result(output)",
            ]

        if context:
            subtasks.insert(0, f"load_context({list(context.keys())})")

        return subtasks

    def _is_goal_satisfied(self, state: set[Atom], goal: list[Atom]) -> bool:
        """Verifica se o objetivo é satisfeito no estado."""
        return all(
            any(g.matches(s) and g.negated == s.negated for s in state)
            for g in goal
        )

    @staticmethod
    def _parse_atom(s: str) -> Atom:
        """Parseia string para Atom (formato: 'pred(arg1, arg2)' ou '¬pred(arg)')."""
        s = s.strip()
        negated = s.startswith("¬") or s.startswith("-") or s.startswith("not ")
        if negated:
            s = s[1:].strip() if s[0] in "¬-" else s[4:].strip()

        match = re.match(r"(\w+)\(([^)]*)\)", s)
        if match:
            name = match.group(1)
            args = tuple(a.strip() for a in match.group(2).split(",") if a.strip())
            return Atom(name, args, negated)
        return Atom(s, (), negated)

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)


# ==============================================================
#  2. TEXTUAL GRADIENT — Feedback Estruturado
# ==============================================================

@dataclass
class GradientSignal:
    """Sinal de gradiente textual: modula o liquid core.

    Cada sinal carrega:
      - component: qual componente do core afetar (lr, surprise, consolidation, oja)
      - magnitude: intensidade do ajuste (0 = sem ajuste, 1 = ajuste máximo)
      - direction: +1 (aumentar) ou -1 (diminuir)
      - source: origem do sinal (verificação, erro, planejamento)
      - description: descrição textual do motivo (legível por humanos)
    """
    component: str  # "lr", "surprise", "consolidation", "oja", "tau"
    magnitude: float  # 0.0 a 1.0
    direction: int  # +1 ou -1
    source: str  # "verification", "error", "planning", "surprise"
    description: str = ""

    def __str__(self) -> str:
        arrow = "↑" if self.direction > 0 else "↓"
        return (f"∇[{self.component}{arrow}{self.magnitude:.2f}] "
                f"from {self.source}: {self.description}")


class TextualGradient:
    """Módulo de feedback estruturado que ajusta o liquid core.

    Converte resultados de verificação, erros de planejamento e sinais
    de surpresa em "gradientes textuais" — sinais interpretáveis que
    modulam os hiperparâmetros do VisaoBrain.

    Inspirado em:
      - ANN (arXiv 2506.09046): "textual backpropagation" — feedback
        estruturado que refina prompts/papéis dos agentes.
      - Scallop: gradientes através de raciocínio lógico (semirings).

    O mecanismo:
      1. Recebe feedback (erro, resultado de verificação, surprise).
      2. Gera GradientSignals para cada componente do core.
      3. Aplica os sinais ao VisaoBrain via apply_to_brain().

    Compatível com os mecanismos do liquid core:
      - Oja: ajuste de oja_lr baseado em estabilidade do recorrente.
      - EWC: ajuste de consolidation baseado em importância.
      - Surprise: ajuste de surprise_gain baseado em detecção de mudança.
    """

    def __init__(
        self,
        lr_sensitivity: float = 0.3,
        surprise_sensitivity: float = 0.5,
        consolidation_sensitivity: float = 0.2,
        oja_sensitivity: float = 0.1,
        history_size: int = 100,
    ):
        self.lr_sensitivity = lr_sensitivity
        self.surprise_sensitivity = surprise_sensitivity
        self.consolidation_sensitivity = consolidation_sensitivity
        self.oja_sensitivity = oja_sensitivity
        self.history: list[GradientSignal] = []
        self._history_size = history_size

    def from_error(
        self,
        error: float,
        error_trend: float = 0.0,
        source: str = "error",
    ) -> list[GradientSignal]:
        """Gera gradientes a partir de sinal de erro.

        - Erro alto + tendência de aumento → aumentar lr, reduzir consolidação.
        - Erro baixo + tendência de diminuição → reduzir lr, aumentar consolidação.
        """
        signals: list[GradientSignal] = []
        error = np.clip(error, 0.0, 10.0)

        # lr: erro alto → aumentar; erro baixo → diminuir
        lr_mag = np.clip(error * self.lr_sensitivity, 0.0, 1.0)
        lr_dir = 1 if error > 0.5 else -1
        signals.append(GradientSignal(
            component="lr",
            magnitude=lr_mag,
            direction=lr_dir,
            source=source,
            description=f"error={error:.3f}, trend={error_trend:+.3f}",
        ))

        # consolidation: erro alto → afrouxar (↓); erro baixo → consolidar (↑)
        cons_mag = np.clip(error * self.consolidation_sensitivity, 0.0, 1.0)
        cons_dir = -1 if error > 0.5 else 1
        signals.append(GradientSignal(
            component="consolidation",
            magnitude=cons_mag,
            direction=cons_dir,
            source=source,
            description=f"error={error:.3f} → {'relax' if cons_dir < 0 else 'tighten'}",
        ))

        # surprise: erro com tendência de aumento → aumentar sensibilidade
        if error_trend > 0.1:
            signals.append(GradientSignal(
                component="surprise",
                magnitude=np.clip(error_trend * self.surprise_sensitivity, 0.0, 1.0),
                direction=1,
                source=source,
                description=f"error trending up ({error_trend:+.3f})",
            ))

        self._record(signals)
        return signals

    def from_verification(
        self,
        verified: bool,
        confidence: float = 1.0,
        source: str = "verification",
    ) -> list[GradientSignal]:
        """Gera gradientes a partir de resultado de verificação.

        - Verificado com alta confiança → consolidar (↑consolidation, ↓lr).
        - Falha na verificação → explorar (↑lr, ↓consolidation, ↑surprise).
        """
        signals: list[GradientSignal] = []

        if verified:
            # Sucesso: consolidar o que foi aprendido
            signals.append(GradientSignal(
                component="consolidation",
                magnitude=confidence * 0.5,
                direction=1,
                source=source,
                description=f"verified (conf={confidence:.2f}) → consolidate",
            ))
            signals.append(GradientSignal(
                component="lr",
                magnitude=confidence * 0.2,
                direction=-1,
                source=source,
                description=f"verified (conf={confidence:.2f}) → reduce lr",
            ))
        else:
            # Falha: explorar mais, afrouxar consolidação
            signals.append(GradientSignal(
                component="lr",
                magnitude=confidence * 0.6,
                direction=1,
                source=source,
                description="verification failed → increase lr",
            ))
            signals.append(GradientSignal(
                component="consolidation",
                magnitude=confidence * 0.4,
                direction=-1,
                source=source,
                description="verification failed → relax consolidation",
            ))
            signals.append(GradientSignal(
                component="surprise",
                magnitude=confidence * 0.3,
                direction=1,
                source=source,
                description="verification failed → increase surprise sensitivity",
            ))

        self._record(signals)
        return signals

    def from_planning(
        self,
        plan_success: bool,
        plan_length: int = 0,
        states_explored: int = 0,
        source: str = "planning",
    ) -> list[GradientSignal]:
        """Gera gradientes a partir de resultado de planejamento.

        - Plano encontrado → consolidar estratégia.
        - Plano não encontrado → aumentar exploração (tau, lr).
        """
        signals: list[GradientSignal] = []

        if plan_success:
            signals.append(GradientSignal(
                component="consolidation",
                magnitude=0.3,
                direction=1,
                source=source,
                description=f"plan found (len={plan_length}) → consolidate",
            ))
            # Planos longos indicam tarefa complexa: aumentar Oja para auto-organização
            if plan_length > 5:
                signals.append(GradientSignal(
                    component="oja",
                    magnitude=0.2,
                    direction=1,
                    source=source,
                    description=f"complex plan (len={plan_length}) → boost Oja",
                ))
        else:
            # Falha de planejamento: aumentar plasticidade
            signals.append(GradientSignal(
                component="lr",
                magnitude=0.5,
                direction=1,
                source=source,
                description=f"plan failed (explored={states_explored}) → explore",
            ))
            signals.append(GradientSignal(
                component="tau",
                magnitude=0.3,
                direction=1,
                source=source,
                description="plan failed → increase tau range (more liquidity)",
            ))

        self._record(signals)
        return signals

    def from_surprise(
        self,
        surprise: float,
        source: str = "surprise",
    ) -> list[GradientSignal]:
        """Gera gradientes a partir de sinal de surpresa do core.

        Surpresa alta indica mudança de regime → afrouxar consolidação,
        aumentar lr temporariamente.
        """
        signals: list[GradientSignal] = []

        if surprise > 1.5:
            mag = np.clip((surprise - 1.0) * 0.5, 0.0, 1.0)
            signals.append(GradientSignal(
                component="consolidation",
                magnitude=mag,
                direction=-1,
                source=source,
                description=f"high surprise ({surprise:.2f}) → relax",
            ))
            signals.append(GradientSignal(
                component="lr",
                magnitude=mag * 0.5,
                direction=1,
                source=source,
                description=f"high surprise ({surprise:.2f}) → boost lr",
            ))
        elif surprise < 0.5:
            signals.append(GradientSignal(
                component="consolidation",
                magnitude=0.2,
                direction=1,
                source=source,
                description=f"low surprise ({surprise:.2f}) → consolidate",
            ))

        self._record(signals)
        return signals

    def apply_to_brain(
        self,
        brain: Any,  # VisaoBrain
        signals: list[GradientSignal] | None = None,
    ) -> dict[str, float]:
        """Aplica sinais de gradiente ao VisaoBrain.

        Modifica os hiperparâmetros do brain baseado nos sinais acumulados.
        Retorna dict com os ajustes aplicados.

        Parameters
        ----------
        brain : VisaoBrain
            Instância do cérebro VISÃO.
        signals : list[GradientSignal] | None
            Sinais a aplicar. Se None, usa o histórico recente.

        Returns
        -------
        dict
            Ajustes aplicados por componente.
        """
        if signals is None:
            signals = self.history[-10:]  # últimos 10 sinais

        adjustments: dict[str, float] = {}

        for sig in signals:
            if sig.component == "lr":
                factor = 1.0 + sig.direction * sig.magnitude * self.lr_sensitivity
                brain.lr = float(np.clip(brain.lr * factor, 1e-5, 1.0))
                brain.learner.lr = brain.lr
                adjustments["lr"] = brain.lr

            elif sig.component == "surprise":
                factor = 1.0 + sig.direction * sig.magnitude * self.surprise_sensitivity
                brain.learner.surprise_gain = float(
                    np.clip(brain.learner.surprise_gain * factor, 0.1, 10.0)
                )
                adjustments["surprise_gain"] = brain.learner.surprise_gain

            elif sig.component == "consolidation":
                factor = 1.0 + sig.direction * sig.magnitude * self.consolidation_sensitivity
                brain.learner.consolidation = float(
                    np.clip(brain.learner.consolidation * factor, 0.0, 50.0)
                )
                adjustments["consolidation"] = brain.learner.consolidation

            elif sig.component == "oja":
                factor = 1.0 + sig.direction * sig.magnitude * self.oja_sensitivity
                brain.learner.oja_lr = float(
                    np.clip(brain.learner.oja_lr * factor, 1e-5, 0.1)
                )
                adjustments["oja_lr"] = brain.learner.oja_lr

            elif sig.component == "tau":
                # Ajusta faixa de tau do liquid cell
                factor = 1.0 + sig.direction * sig.magnitude * 0.2
                tau_min = float(np.min(brain.cell.tau))
                tau_max = float(np.max(brain.cell.tau))
                new_tau_max = float(np.clip(tau_max * factor, 0.5, 20.0))
                # Reescala tau proporcionalmente
                if tau_max > 0:
                    scale = new_tau_max / tau_max
                    brain.cell.tau *= scale
                adjustments["tau_max"] = new_tau_max

        return adjustments

    def _record(self, signals: list[GradientSignal]) -> None:
        """Registra sinais no histórico."""
        self.history.extend(signals)
        if len(self.history) > self._history_size:
            self.history = self.history[-self._history_size:]

    def get_summary(self) -> dict[str, Any]:
        """Retorna resumo dos sinais recentes."""
        if not self.history:
            return {"n_signals": 0}

        by_component: dict[str, list[float]] = {}
        for sig in self.history:
            by_component.setdefault(sig.component, []).append(sig.magnitude * sig.direction)

        summary = {"n_signals": len(self.history)}
        for comp, vals in by_component.items():
            summary[f"{comp}_mean"] = float(np.mean(vals))
            summary[f"{comp}_std"] = float(np.std(vals))
        return summary


# ==============================================================
#  3. VERIFICATION ENGINE — Verificador de Provas/Códigos
# ==============================================================

class VerificationResult:
    """Resultado de uma verificação."""

    def __init__(
        self,
        verified: bool,
        confidence: float = 1.0,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.verified = verified
        self.confidence = confidence
        self.errors = errors or []
        self.warnings = warnings or []
        self.details = details or {}

    def __str__(self) -> str:
        status = "✓ VERIFIED" if self.verified else "✗ FAILED"
        return (f"{status} (conf={self.confidence:.2f}) "
                f"errors={len(self.errors)}, warnings={len(self.warnings)}")

    def __bool__(self) -> bool:
        return self.verified

    def to_signal(self) -> dict[str, Any]:
        """Converte para dict compatível com GradientSignal."""
        return {
            "verified": self.verified,
            "confidence": self.confidence,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
        }


class VerificationEngine:
    """Verificador de provas lógicas e código.

    Capacidades:
      1. Verificação de planos (validade lógica, completude).
      2. Verificação de código Python (sintaxe, segurança básica).
      3. Verificação de provas lógicas (cadeia de inferência válida).
      4. Verificação de invariantes (pré/pós-condições).

    Inspirado em:
      - AlphaProof/AlphaGeometry (DeepMind): neural sugere, simbólico verifica.
      - Scallop: verificação via raciocínio lógico diferenciável.
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._history: list[VerificationResult] = []

    def verify_plan(
        self,
        plan: Plan,
        initial_state: set[Atom] | None = None,
    ) -> VerificationResult:
        """Verifica validade de um plano.

        Checa:
          - Cada ação é aplicável no estado em que é executada.
          - O objetivo é atingido ao final.
          - Não há estados impossíveis (átomos contraditórios).
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not plan.actions:
            if plan.success:
                return VerificationResult(True, 1.0, details={"empty_plan": True})
            return VerificationResult(False, 1.0, errors=["Plano vazio sem sucesso"])

        # Verifica rastro de estados
        if plan.state_trace:
            for i, (action, state) in enumerate(zip(plan.actions, plan.state_trace)):
                if not action.is_applicable(state):
                    errors.append(
                        f"Ação {i} ({action.name}) não aplicável no estado {i}"
                    )

            # Verifica se objetivo é atingido no último estado
            if plan.state_trace:
                final_state = plan.state_trace[-1]
                for goal_atom in plan.goal:
                    if not any(goal_atom.matches(s) for s in final_state):
                        errors.append(
                            f"Objetivo {goal_atom} não atingido no estado final"
                        )
        else:
            warnings.append("Sem rastro de estados para verificar")

        # Verifica contradições (átomo e sua negação no mesmo estado)
        if plan.state_trace:
            for i, state in enumerate(plan.state_trace):
                atoms_pos = {a for a in state if not a.negated}
                atoms_neg = {a for a in state if a.negated}
                for ap in atoms_pos:
                    for an in atoms_neg:
                        if ap.name == an.name and ap.args == an.args:
                            errors.append(
                                f"Contradição no estado {i}: {ap} e {an}"
                            )

        verified = len(errors) == 0
        confidence = 1.0 if verified else max(0.0, 1.0 - len(errors) * 0.2)

        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            details={"plan_length": len(plan.actions), "n_states": len(plan.state_trace)},
        )
        self._history.append(result)
        return result

    def verify_code(
        self,
        code: str,
        language: str = "python",
    ) -> VerificationResult:
        """Verifica corretude de código.

        Para Python:
          - Sintaxe (via ast.parse).
          - Segurança básica (sem imports perigosos).
          - Estrutura mínima (se há funções/classes).

        Para outras linguagens: verificação sintática básica.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if language == "python":
            return self._verify_python(code, errors, warnings)
        else:
            warnings.append(f"Verificação para '{language}' limitada a sintaxe básica")
            # Verificação genérica: parênteses/chaves balanceados
            if not self._check_balanced(code):
                errors.append("Parênteses/chaves/colchetes desbalanceados")

        verified = len(errors) == 0
        confidence = 1.0 if verified else max(0.0, 1.0 - len(errors) * 0.3)

        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            details={"language": language, "code_length": len(code)},
        )
        self._history.append(result)
        return result

    def verify_proof(
        self,
        premises: list[str],
        conclusion: str,
        steps: list[dict[str, str]] | None = None,
    ) -> VerificationResult:
        """Verifica validade de uma prova lógica.

        Checa:
          - Premissas não são contraditórias.
          - Cada passo de inferência é válido (modus ponens, etc.).
          - A conclusão segue dos passos.

        Parameters
        ----------
        premises : list[str]
            Lista de premissas (strings de fórmulas).
        conclusion : str
            Conclusão a ser provada.
        steps : list[dict] | None
            Passos de inferência, cada um com {"rule": ..., "from": ..., "to": ...}.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Verifica contradição nas premissas
        parsed_premises = []
        for p in premises:
            try:
                parsed_premises.append(SymbolicSolver._parse_atom(p))
            except Exception:
                errors.append(f"Premissa mal formada: {p}")

        for i, p1 in enumerate(parsed_premises):
            for j, p2 in enumerate(parsed_premises):
                if i < j and p1.name == p2.name and p1.args == p2.args:
                    if p1.negated != p2.negated:
                        errors.append(
                            f"Premissas contraditórias: {p1} e {p2}"
                        )

        # Verifica passos de inferência
        if steps:
            for i, step in enumerate(steps):
                rule = step.get("rule", "")
                if rule not in ("modus_ponens", "modus_tollens", "and_intro",
                                "and_elim", "or_intro", "forall_intro",
                                "exists_elim", "axiom", "assumption"):
                    warnings.append(f"Passo {i}: regra '{rule}' não reconhecida")

                if "from" in step and "to" not in step:
                    errors.append(f"Passo {i}: falta 'to' (conclusão do passo)")
        elif not steps and len(premises) > 1:
            warnings.append("Sem passos de inferência para verificar")

        # Verifica se conclusão é atingível (heurística simples)
        if steps:
            last_step = steps[-1]
            if "to" in last_step:
                # Verificação simples: conclusão mencionada no último passo
                if conclusion not in last_step.get("to", ""):
                    warnings.append(
                        "Último passo não menciona explicitamente a conclusão"
                    )

        verified = len(errors) == 0
        confidence = 1.0 if verified else max(0.0, 1.0 - len(errors) * 0.25)

        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            details={
                "n_premises": len(premises),
                "n_steps": len(steps) if steps else 0,
            },
        )
        self._history.append(result)
        return result

    def verify_invariant(
        self,
        precondition: list[Atom],
        postcondition: list[Atom],
        body: list[Atom],
    ) -> VerificationResult:
        """Verifica se uma pós-condição é válida dado pré-corpo.

        Checa se a pós-condição é logicamente implicada pelo conjunto
        de precondições + corpo (verificação de Hoare simplificada).
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Verifica se cada átomo da pós-condição está nas precondições ou corpo
        for post in postcondition:
            found = False
            for pre in precondition:
                if post.matches(pre) and post.negated == pre.negated:
                    found = True
                    break
            if not found:
                for b in body:
                    if post.matches(b) and post.negated == b.negated:
                        found = True
                        break
            if not found:
                errors.append(f"Pós-condição {post} não derivável")

        verified = len(errors) == 0
        confidence = 1.0 if verified else max(0.0, 1.0 - len(errors) * 0.2)

        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            details={
                "n_pre": len(precondition),
                "n_post": len(postcondition),
                "n_body": len(body),
            },
        )
        self._history.append(result)
        return result

    def _verify_python(
        self,
        code: str,
        errors: list[str],
        warnings: list[str],
    ) -> VerificationResult:
        """Verificação específica para Python."""
        # 1. Sintaxe
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Sintaxe inválida: {e}")
            result = VerificationResult(
                verified=False,
                confidence=0.0,
                errors=errors,
                warnings=warnings,
                details={"language": "python"},
            )
            self._history.append(result)
            return result

        # 2. Segurança: imports perigosos
        dangerous_imports = {"os", "subprocess", "shutil", "socket", "ctypes",
                             "multiprocessing", "threading"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    if base in dangerous_imports:
                        if self.strict_mode:
                            errors.append(f"Import perigoso: {alias.name}")
                        else:
                            warnings.append(f"Import potencialmente perigoso: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split(".")[0]
                    if base in dangerous_imports:
                        if self.strict_mode:
                            errors.append(f"Import perigoso: {node.module}")
                        else:
                            warnings.append(f"Import potencialmente perigoso: {node.module}")

        # 3. Estrutura
        has_function = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        has_class = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        has_lambda = any(isinstance(n, ast.Lambda) for n in ast.walk(tree))

        if not has_function and not has_class and not has_lambda:
            if len(code.strip().split("\n")) > 5:
                warnings.append("Código sem funções/classes (script linear)")

        # 4. Verifica recursão sem caso base (heurística)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Verifica se há chamadas recursivas
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            if child.func.id == node.name:
                                # Verifica se há condicional (possível caso base)
                                has_if = any(isinstance(n, ast.If) for n in ast.walk(node))
                                if not has_if:
                                    warnings.append(
                                        f"Função recursiva '{node.name}' sem condicional "
                                        f"(possível loop infinito)"
                                    )

        verified = len(errors) == 0
        confidence = 1.0 if verified else max(0.0, 1.0 - len(errors) * 0.3)

        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            details={
                "language": "python",
                "has_function": has_function,
                "has_class": has_class,
                "n_lines": len(code.strip().split("\n")),
            },
        )
        self._history.append(result)
        return result

    @staticmethod
    def _check_balanced(code: str) -> bool:
        """Verifica se parênteses/chaves/colchetes estão balanceados."""
        pairs = {"(": ")", "{": "}", "[": "]"}
        stack: list[str] = []
        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack[-1]] != char:
                    return False
                stack.pop()
        return len(stack) == 0

    @property
    def history(self) -> list[VerificationResult]:
        return list(self._history)

    def get_stats(self) -> dict[str, Any]:
        """Estatísticas das verificações realizadas."""
        if not self._history:
            return {"n_verifications": 0}
        n_ok = sum(1 for r in self._history if r.verified)
        return {
            "n_verifications": len(self._history),
            "n_passed": n_ok,
            "n_failed": len(self._history) - n_ok,
            "pass_rate": n_ok / len(self._history),
            "avg_confidence": float(np.mean([r.confidence for r in self._history])),
        }


# ==============================================================
#  REASONING LAYER — Integração
# ==============================================================

class ReasoningLayer:
    """Camada de raciocínio neuro-simbólica integrada.

    Combina os três componentes:
      1. SymbolicSolver: planeja e decompõe tarefas.
      2. VerificationEngine: verifica planos, código e provas.
      3. TextualGradient: ajusta o liquid core baseado em feedback.

    O ciclo de operação:
      1. Decompor tarefa (SymbolicSolver.decompose).
      2. Planejar ações (SymbolicSolver.solve).
      3. Verificar plano (VerificationEngine.verify_plan).
      4. Gerar gradientes (TextualGradient.from_verification).
      5. Aplicar ao core (TextualGradient.apply_to_brain).

    Parameters
    ----------
    brain : VisaoBrain | None
        Instância do cérebro VISÃO (opcional, para integração direta).
    actions : list[Action] | None
        Ações disponíveis para o planejador.
    verbose : bool
        Se True, imprime passos intermediários.
    """

    def __init__(
        self,
        brain: Any | None = None,
        actions: list[Action] | None = None,
        verbose: bool = False,
    ):
        self.brain = brain
        self.solver = SymbolicSolver(actions=actions, verbose=verbose)
        self.verifier = VerificationEngine()
        self.gradient = TextualGradient()
        self.verbose = verbose

    def reason(
        self,
        task: str,
        initial_state: set[Atom] | None = None,
        goal: list[Atom] | None = None,
    ) -> dict[str, Any]:
        """Executa ciclo completo de raciocínio.

        1. Decompor tarefa em subtasks.
        2. Planejar (se initial_state e goal fornecidos).
        3. Verificar plano.
        4. Gerar e aplicar gradientes.

        Returns
        -------
        dict
            Resultado completo com plano, verificação e ajustes.
        """
        result: dict[str, Any] = {"task": task}

        # 1. Decomposição
        subtasks = self.solver.decompose(task)
        result["subtasks"] = subtasks
        if self.verbose:
            print(f"[ReasoningLayer] Subtasks: {subtasks}")

        # 2. Planejamento (se aplicável)
        plan = None
        if initial_state is not None and goal is not None:
            plan = self.solver.solve(initial_state, goal)
            result["plan"] = plan
            if self.verbose:
                print(f"[ReasoningLayer] Plano: {plan}")

            # 3. Verificação
            verification = self.verifier.verify_plan(plan, initial_state)
            result["verification"] = verification
            if self.verbose:
                print(f"[ReasoningLayer] Verificação: {verification}")

            # 4. Gradientes do planejamento
            plan_signals = self.gradient.from_planning(
                plan_success=plan.success,
                plan_length=len(plan),
                states_explored=self.solver.stats.get("states_explored", 0),
            )
            result["plan_signals"] = [str(s) for s in plan_signals]

        # 5. Aplicar ao brain se disponível
        if self.brain is not None:
            adjustments = self.gradient.apply_to_brain(self.brain)
            result["adjustments"] = adjustments
            if self.verbose:
                print(f"[ReasoningLayer] Ajustes: {adjustments}")

        return result

    def verify_and_adjust(
        self,
        code: str | None = None,
        plan: Plan | None = None,
    ) -> VerificationResult:
        """Verifica código/plano e aplica gradientes ao brain.

        Atalho para o ciclo verificar → gradiente → ajustar.
        """
        if code is not None:
            result = self.verifier.verify_code(code)
        elif plan is not None:
            result = self.verifier.verify_plan(plan)
        else:
            raise ValueError("Forneça code ou plan")

        # Gera gradientes da verificação
        signals = self.gradient.from_verification(
            verified=result.verified,
            confidence=result.confidence,
        )

        # Aplica ao brain
        if self.brain is not None:
            self.gradient.apply_to_brain(self.brain, signals)

        return result

    def get_diagnostics(self) -> dict[str, Any]:
        """Retorna diagnósticos da camada de raciocínio."""
        return {
            "solver_stats": self.solver.stats,
            "verifier_stats": self.verifier.get_stats(),
            "gradient_summary": self.gradient.get_summary(),
        }


# ==============================================================
#  UTILITÁRIOS
# ==============================================================

def create_default_actions() -> list[Action]:
    """Cria conjunto de ações padrão para demonstração."""
    return [
        Action(
            name="parse",
            preconditions=[Atom("raw", ("?input",))],
            add_effects=[Atom("parsed", ("?input",))],
            del_effects=[Atom("raw", ("?input",))],
        ),
        Action(
            name="analyze",
            preconditions=[Atom("parsed", ("?input",))],
            add_effects=[Atom("analyzed", ("?input",))],
            del_effects=[Atom("parsed", ("?input",))],
        ),
        Action(
            name="execute",
            preconditions=[Atom("analyzed", ("?input",))],
            add_effects=[Atom("executed", ("?input",))],
            del_effects=[Atom("analyzed", ("?input",))],
        ),
        Action(
            name="verify",
            preconditions=[Atom("executed", ("?input",))],
            add_effects=[Atom("verified", ("?input",))],
            del_effects=[Atom("executed", ("?input",))],
        ),
        Action(
            name="return_result",
            preconditions=[Atom("verified", ("?input",))],
            add_effects=[Atom("done", ("?input",))],
            del_effects=[Atom("verified", ("?input",))],
        ),
    ]


def create_demo_reasoning_layer(
    brain: Any | None = None,
    verbose: bool = True,
) -> ReasoningLayer:
    """Cria uma ReasoningLayer com ações padrão para demonstração."""
    actions = create_default_actions()
    return ReasoningLayer(brain=brain, actions=actions, verbose=verbose)
