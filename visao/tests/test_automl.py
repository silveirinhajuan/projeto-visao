"""Testes TDD da tarefa 5.7 — Auto-ajuste local em sandbox (L3 AutoML).

O AutoTuner propõe combinações de hiperparâmetros (τ, taxas de regra local,
topologia), avalia cada uma DENTRO do Sandbox (sem rede, sem persistência) e
devolve a melhor — tudo registrado no Archive. Nada de self-rewrite em rede:
o código de cada variante é um marcador neutro, fora de qualquer zona proibida
(R3). O eval script é fornecido por quem chama e decide o que medir.
"""
from __future__ import annotations

from visao.evolve.automl import AutoTuner
from visao.evolve.archive import Archive
from visao.evolve.sandbox import Sandbox
from visao.evolve.mutate import _contains_forbidden


def _tuner(search_space, mode="grid", seed=0, base_config=None, budget=None, sandbox=None):
    return AutoTuner(
        archive=Archive(),
        sandbox=sandbox or Sandbox(),
        search_space=search_space,
        base_config=base_config or {},
        mode=mode,
        seed=seed,
        budget=budget,
    )


def test_tuner_evaluates_and_returns_best():
    # eval retorna o próprio tau; o melhor deve ser o maior
    tuner = _tuner({"tau": [0.1, 0.5, 0.9]})
    best = tuner.run('report_fitness(variant["weights"]["tau"])')
    assert best is not None
    cfg = tuner.archive.get(best).weights
    assert cfg["tau"] == 0.9


def test_tuner_respects_budget():
    # 2 params x 3 valores = 9 combinações; orçamento 4 => só 4 avaliadas
    tuner = _tuner(
        {"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]},
        budget=4,
    )
    tuner.run('report_fitness(0.5)')
    assert len(list(tuner.archive._nodes)) == 4
    for vid in tuner.archive._nodes:
        assert tuner.archive.get(vid).fitness is not None


def test_tuner_deterministic_with_seed():
    space = {"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.0, 2.0]}
    eval_script = 'report_fitness(variant["weights"]["a"] + variant["weights"]["b"])'

    t1 = _tuner(space, mode="random", seed=42, budget=10)
    t2 = _tuner(space, mode="random", seed=42, budget=10)
    b1 = t1.run(eval_script)
    b2 = t2.run(eval_script)

    assert t1.archive.get(b1).weights == t2.archive.get(b2).weights


def test_tuner_variant_code_is_r3_safe():
    # nenhuma variante produzida pelo tuner pode tocar zona proibida (R3)
    tuner = _tuner({"tau": [0.1, 0.5], "oja_lr": [0.01, 0.05]})
    tuner.run('report_fitness(1.0)')
    assert len(tuner.archive._nodes) > 0
    for vid in tuner.archive._nodes:
        code = tuner.archive.get(vid).code
        assert not _contains_forbidden(code)


def test_tuner_records_all_configs_in_archive():
    space = {"tau": [0.1, 0.2], "oja_lr": [0.01, 0.02]}
    tuner = _tuner(space)
    tuner.run('report_fitness(0.3)')

    seen = [tuple(sorted(v.weights.items())) for v in tuner.archive._nodes.values()]
    # 2 x 2 combinações distintas
    assert len(seen) == 4
    assert len(set(seen)) == 4
    for vid in tuner.archive._nodes:
        assert tuner.archive.get(vid).fitness is not None
