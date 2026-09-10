"""
test_neuro_symbolic.py — Teste simples do Reasoning Layer (Tarefa 14.0).

Testa:
  1. SymbolicSolver: planejamento com ações STRIPS-like.
  2. TextualGradient: geração de sinais de feedback.
  3. VerificationEngine: verificação de código e planos.
  4. ReasoningLayer: integração com VisaoBrain.
"""

import sys
from pathlib import Path

import numpy as np

# Garantir paths
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from visao.reasoning.neuro_symbolic import (
    Action,
    Atom,
    Plan,
    ReasoningLayer,
    SymbolicSolver,
    TextualGradient,
    VerificationEngine,
    create_default_actions,
    create_demo_reasoning_layer,
)
from visao.brain import VisaoBrain


def test_symbolic_solver():
    """Testa planejamento lógico."""
    print("=" * 60)
    print("TESTE 1: SymbolicSolver")
    print("=" * 60)

    # Criar ações
    actions = create_default_actions()
    solver = SymbolicSolver(actions=actions, verbose=True)

    # Definir estado inicial e objetivo
    initial_state = {Atom("raw", ("task1",))}
    goal = [Atom("done", ("task1",))]

    # Resolver
    plan = solver.solve(initial_state, goal)
    print(f"Plano: {plan}")
    print(f"Estatísticas: {solver.stats}")

    assert plan.success, "Plano deveria ser encontrado"
    assert len(plan.actions) == 5, f"Plano deveria ter 5 ações, tem {len(plan.actions)}"
    print("✓ SymbolicSolver: PASS\n")


def test_textual_gradient():
    """Testa geração de gradientes textuais."""
    print("=" * 60)
    print("TESTE 2: TextualGradient")
    print("=" * 60)

    tg = TextualGradient()

    # Testar a partir de erro
    signals = tg.from_error(error=0.8, error_trend=0.2)
    print(f"Sinais de erro: {[str(s) for s in signals]}")

    # Testar a partir de verificação
    signals = tg.from_verification(verified=False, confidence=0.9)
    print(f"Sinais de verificação: {[str(s) for s in signals]}")

    # Testar a partir de planejamento
    signals = tg.from_planning(plan_success=True, plan_length=7)
    print(f"Sinais de planejamento: {[str(s) for s in signals]}")

    # Testar aplicação ao brain
    brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
    initial_lr = brain.lr
    adjustments = tg.apply_to_brain(brain)
    print(f"Ajustes aplicados: {adjustments}")
    print(f"lr: {initial_lr} → {brain.lr}")

    summary = tg.get_summary()
    print(f"Resumo: {summary}")
    print("✓ TextualGradient: PASS\n")


def test_verification_engine():
    """Testa verificador de código e planos."""
    print("=" * 60)
    print("TESTE 3: VerificationEngine")
    print("=" * 60)

    verifier = VerificationEngine()

    # Testar código válido
    code_ok = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    result = verifier.verify_code(code_ok)
    print(f"Código válido: {result}")
    assert result.verified, "Código válido deveria passar"

    # Testar código com sintaxe inválida
    code_bad = """
def broken(
    return 1 +
"""
    result = verifier.verify_code(code_bad)
    print(f"Código inválido: {result}")
    assert not result.verified, "Código inválido deveria falhar"

    # Testar código com import perigoso
    code_danger = """
import os
os.system("rm -rf /")
"""
    result = verifier.verify_code(code_danger)
    print(f"Código perigoso: {result}")
    assert len(result.warnings) > 0, "Deveria gerar warning de import perigoso"

    # Testar verificação de prova
    result = verifier.verify_proof(
        premises=["P(a)", "P(a) → Q(a)"],
        conclusion="Q(a)",
        steps=[
            {"rule": "assumption", "from": "", "to": "P(a)"},
            {"rule": "assumption", "from": "", "to": "P(a) → Q(a)"},
            {"rule": "modus_ponens", "from": "P(a), P(a) → Q(a)", "to": "Q(a)"},
        ],
    )
    print(f"Prova: {result}")
    assert result.verified, "Prova válida deveria passar"

    # Testar verificação de plano
    actions = create_default_actions()
    solver = SymbolicSolver(actions=actions)
    initial = {Atom("raw", ("t",))}
    goal = [Atom("done", ("t",))]
    plan = solver.solve(initial, goal)
    result = verifier.verify_plan(plan, initial)
    print(f"Plano: {result}")
    assert result.verified, "Plano válido deveria passar"

    stats = verifier.get_stats()
    print(f"Estatísticas: {stats}")
    print("✓ VerificationEngine: PASS\n")


def test_reasoning_layer_integration():
    """Testa integração completa com VisaoBrain."""
    print("=" * 60)
    print("TESTE 4: ReasoningLayer + VisaoBrain")
    print("=" * 60)

    # Criar brain e reasoning layer
    brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
    layer = create_demo_reasoning_layer(brain=brain, verbose=True)

    # Executar raciocínio
    result = layer.reason("calcular fibonacci de 10")
    print(f"\nResultado: {result.keys()}")
    print(f"Subtasks: {result['subtasks']}")

    # Verificar código e ajustar brain
    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
    verification = layer.verify_and_adjust(code=code)
    print(f"Verificação: {verification}")

    # Verificar plano
    initial = {Atom("raw", ("task",))}
    goal = [Atom("done", ("task",))]
    plan = layer.solver.solve(initial, goal)
    verification = layer.verify_and_adjust(plan=plan)
    print(f"Verificação do plano: {verification}")

    # Diagnósticos
    diagnostics = layer.get_diagnostics()
    print(f"Diagnósticos: {diagnostics}")
    print("✓ ReasoningLayer + VisaoBrain: PASS\n")


def test_decomposition():
    """Testa decomposição de tarefas."""
    print("=" * 60)
    print("TESTE 5: Decomposição de Tarefas")
    print("=" * 60)

    solver = SymbolicSolver()

    tasks = [
        "calcular fibonacci de 10",
        "provar que todo número par > 2 é soma de dois primos",
        "verificar se o código está correto",
        "buscar todos os registros com valor > 100",
        "tarefa desconhecida xyz",
    ]

    for task in tasks:
        subtasks = solver.decompose(task)
        print(f"  {task}: {subtasks}")

    print("✓ Decomposição: PASS\n")


def test_gradient_compatibility():
    """Testa compatibilidade dos gradientes com Oja + EWC + Surprise."""
    print("=" * 60)
    print("TESTE 6: Compatibilidade com Oja + EWC + Surprise")
    print("=" * 60)

    brain = VisaoBrain(
        n_in=2, n_hidden=16, n_out=1,
        consolidation=8.0, meta_learn=True, seed=42
    )
    tg = TextualGradient()

    # Simular aprendizado com ajustes do reasoning layer
    rng = np.random.default_rng(42)
    initial_lr = brain.lr
    initial_consolidation = brain.learner.consolidation
    initial_surprise_gain = brain.learner.surprise_gain

    print(f"lr inicial: {initial_lr}")
    print(f"consolidation inicial: {initial_consolidation}")
    print(f"surprise_gain inicial: {initial_surprise_gain}")

    # Simular 10 passos de aprendizado com feedback
    for i in range(10):
        x = rng.normal(0, 1, 2)
        y = np.array([np.sin(x[0])])

        result = brain.learn(x, y)

        # Gerar gradientes baseados no erro
        signals = tg.from_error(
            error=result["err"],
            error_trend=result["surprise"] - 1.0,
        )
        tg.apply_to_brain(brain, signals)

    print(f"\nApós 10 passos:")
    print(f"  lr: {brain.lr:.6f} (era {initial_lr})")
    print(f"  consolidation: {brain.learner.consolidation:.4f} (era {initial_consolidation})")
    print(f"  surprise_gain: {brain.learner.surprise_gain:.4f} (era {initial_surprise_gain})")
    print(f"  oja_lr: {brain.learner.oja_lr:.6f}")

    # Verificar que os mecanismos estão ativos
    assert brain.learner.oja_lr > 0, "Oja deveria estar ativo"
    assert brain.learner.consolidation > 0, "EWC deveria estar ativo"
    assert brain.learner.surprise_gain > 0, "Surprise deveria estar ativo"

    print("✓ Compatibilidade: PASS\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTE DO REASONING LAYER — Tarefa 14.0")
    print("=" * 60 + "\n")

    test_symbolic_solver()
    test_textual_gradient()
    test_verification_engine()
    test_reasoning_layer_integration()
    test_decomposition()
    test_gradient_compatibility()

    print("=" * 60)
    print("TODOS OS TESTES PASSARAM ✓")
    print("=" * 60)
