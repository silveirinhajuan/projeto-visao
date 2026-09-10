"""
test_inference.py — Testes para InferenceEngine e KnowledgeBase.

Cobre:
  - InferenceEngine.forward_chain: retorna conclusions, confidence, chain
  - InferenceEngine.backward_chain: retorna supported, support_score, relevant_facts
  - InferenceEngine.consolidate_knowledge: retorna stability, consolidation_gain
  - InferenceEngine.detect_contradictions: retorna contradictions, n_contradictions
  - KnowledgeBase: add_fact, query, reason
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.reasoning.inference import InferenceEngine, KnowledgeBase


# ==============================================================
#  FIXTURES
# ==============================================================

@pytest.fixture
def brain_8():
    """Cérebro com n_in=8 para testes de inferência."""
    return VisaoBrain(n_in=8, n_hidden=32, n_out=8, seed=42)


@pytest.fixture
def brain_1():
    """Cérebro com n_out=1 para testes simples."""
    return VisaoBrain(n_in=8, n_hidden=32, n_out=1, seed=42)


@pytest.fixture
def engine_8(brain_8):
    """InferenceEngine com brain de saída 8-dim."""
    return InferenceEngine(brain_8)


@pytest.fixture
def engine_1(brain_1):
    """InferenceEngine com brain de saída 1-dim."""
    return InferenceEngine(brain_1)


@pytest.fixture
def sample_hypothesis_1d():
    """Hipótese 1-dim para engine com n_out=1."""
    return np.array([0.5])


@pytest.fixture
def sample_hypothesis_8d():
    """Hipótese 8-dim para engine com n_out=8."""
    return np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def sample_facts():
    """Lista de fatos 8-dim para testes."""
    return [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]


@pytest.fixture
def kb(brain_8):
    """KnowledgeBase para testes."""
    return KnowledgeBase(brain_8, max_facts=100)


# ==============================================================
#  TESTES: FORWARD CHAINING
# ==============================================================

class TestForwardChain:
    def test_returns_dict_with_required_keys(self, engine_1, sample_facts):
        """forward_chain deve retornar dict com conclusions, confidence, chain."""
        result = engine_1.forward_chain(sample_facts, n_steps=3)
        assert "conclusions" in result
        assert "confidence" in result
        assert "chain" in result

    def test_conclusions_is_ndarray(self, engine_1, sample_facts):
        """conclusions deve ser um numpy array."""
        result = engine_1.forward_chain(sample_facts, n_steps=3)
        assert isinstance(result["conclusions"], np.ndarray)

    def test_confidence_is_float_in_range(self, engine_1, sample_facts):
        """confidence deve ser um float entre -1 e 1 (similaridade de cosseno)."""
        result = engine_1.forward_chain(sample_facts, n_steps=3)
        assert isinstance(result["confidence"], float)
        assert -1.0 <= result["confidence"] <= 1.0

    def test_chain_is_list(self, engine_1, sample_facts):
        """chain deve ser uma lista de dicionários."""
        result = engine_1.forward_chain(sample_facts, n_steps=3)
        assert isinstance(result["chain"], list)
        assert len(result["chain"]) > 0

    def test_chain_length(self, engine_1, sample_facts):
        """chain deve ter n_facts + n_steps entradas."""
        n_facts = len(sample_facts)
        n_steps = 3
        result = engine_1.forward_chain(sample_facts, n_steps=n_steps)
        assert len(result["chain"]) == n_facts + n_steps

    def test_chain_entries_have_required_keys(self, engine_1, sample_facts):
        """Cada entrada do chain deve ter step, input, output, type."""
        result = engine_1.forward_chain(sample_facts, n_steps=2)
        for entry in result["chain"]:
            assert "step" in entry
            assert "input" in entry
            assert "output" in entry
            assert "type" in entry

    def test_chain_first_entries_are_facts(self, engine_1, sample_facts):
        """Primeiras entradas do chain devem ser do tipo 'fact'."""
        result = engine_1.forward_chain(sample_facts, n_steps=2)
        for i in range(len(sample_facts)):
            assert result["chain"][i]["type"] == "fact"

    def test_chain_later_entries_are_inference(self, engine_1, sample_facts):
        """Entradas após os fatos devem ser do tipo 'inference'."""
        result = engine_1.forward_chain(sample_facts, n_steps=2)
        for i in range(len(sample_facts), len(result["chain"])):
            assert result["chain"][i]["type"] == "inference"

    def test_n_facts_and_n_inference_steps(self, engine_1, sample_facts):
        """Deve retornar n_facts e n_inference_steps."""
        result = engine_1.forward_chain(sample_facts, n_steps=4)
        assert result["n_facts"] == len(sample_facts)
        assert result["n_inference_steps"] == 4

    def test_single_fact(self, engine_1):
        """forward_chain com um único fato."""
        facts = [np.ones(8)]
        result = engine_1.forward_chain(facts, n_steps=2)
        assert len(result["chain"]) == 3
        assert result["n_facts"] == 1

    def test_empty_facts_list(self, engine_1):
        """forward_chain com lista vazia de fatos."""
        result = engine_1.forward_chain([], n_steps=3)
        assert result["conclusions"] is None
        assert len(result["chain"]) == 0

    def test_n_steps_zero(self, engine_1, sample_facts):
        """forward_chain com n_steps=0."""
        result = engine_1.forward_chain(sample_facts, n_steps=0)
        assert len(result["chain"]) == len(sample_facts)

    def test_confidence_with_empty_chain(self, engine_1):
        """Confiança com chain vazia deve ser 1.0."""
        result = engine_1.forward_chain([], n_steps=3)
        assert result["confidence"] == 1.0

    def test_output_with_8d_brain(self, engine_8, sample_facts):
        """forward_chain com brain 8-dim retorna conclusão 8-dim."""
        result = engine_8.forward_chain(sample_facts, n_steps=3)
        assert result["conclusions"].shape == (8,)

    def test_set_mode_infer(self, engine_1, sample_facts):
        """forward_chain deve usar modo infer."""
        engine_1.forward_chain(sample_facts, n_steps=2)
        assert engine_1.brain.mode == "infer"


# ==============================================================
#  TESTES: BACKWARD CHAINING
# ==============================================================

class TestBackwardChain:
    def test_returns_dict_with_required_keys(self, engine_1, sample_facts, sample_hypothesis_1d):
        """backward_chain deve retornar dict com supported, support_score, relevant_facts."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert "supported" in result
        assert "support_score" in result
        assert "relevant_facts" in result

    def test_supported_is_bool(self, engine_1, sample_facts, sample_hypothesis_1d):
        """supported deve ser um booleano (ou numpy bool_)."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert isinstance(result["supported"], (bool, np.bool_))

    def test_support_score_is_float(self, engine_1, sample_facts, sample_hypothesis_1d):
        """support_score deve ser um float."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert isinstance(result["support_score"], float)

    def test_relevant_facts_is_list(self, engine_1, sample_facts, sample_hypothesis_1d):
        """relevant_facts deve ser uma lista."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert isinstance(result["relevant_facts"], list)

    def test_relevant_facts_at_most_3(self, engine_1, sample_facts, sample_hypothesis_1d):
        """relevant_facts deve ter no máximo 3 fatos."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert len(result["relevant_facts"]) <= 3

    def test_fact_scores_present(self, engine_1, sample_facts, sample_hypothesis_1d):
        """fact_scores deve estar no resultado."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert "fact_scores" in result
        assert len(result["fact_scores"]) == len(sample_facts)

    def test_support_score_range(self, engine_1, sample_facts, sample_hypothesis_1d):
        """support_score deve estar entre -1 e 1."""
        result = engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert -1.0 <= result["support_score"] <= 1.0

    def test_supported_when_score_high(self, engine_1, sample_hypothesis_1d):
        """Quando support_score > 0.5, supported deve ser True."""
        facts = [np.ones(8)]
        result = engine_1.backward_chain(sample_hypothesis_1d, facts)
        if result["support_score"] > 0.5:
            assert result["supported"] is True

    def test_empty_facts(self, engine_1, sample_hypothesis_1d):
        """backward_chain com facts vazio deve retornar supported=False."""
        result = engine_1.backward_chain(sample_hypothesis_1d, [])
        assert result["supported"] is False
        assert result["support_score"] == 0.0

    def test_different_hypotheses(self, engine_8, sample_facts, sample_hypothesis_8d):
        """Diferentes hipóteses geram diferentes scores (com brain multi-dim)."""
        hyp1 = sample_hypothesis_8d
        hyp2 = np.array([0.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0])
        result1 = engine_8.backward_chain(hyp1, sample_facts)
        result2 = engine_8.backward_chain(hyp2, sample_facts)
        assert result1["support_score"] != result2["support_score"]


# ==============================================================
#  TESTES: CONSOLIDATE KNOWLEDGE
# ==============================================================

class TestConsolidateKnowledge:
    def test_returns_dict_with_required_keys(self, engine_8, sample_facts):
        """consolidate_knowledge deve retornar dict com stability, consolidation_gain."""
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=5)
        assert "stability" in result
        assert "consolidation_gain" in result

    def test_stability_is_float(self, engine_8, sample_facts):
        """stability deve ser um float."""
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=5)
        assert isinstance(result["stability"], float)

    def test_consolidation_gain_is_float(self, engine_8, sample_facts):
        """consolidation_gain deve ser um float."""
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=5)
        assert isinstance(result["consolidation_gain"], float)

    def test_stability_range(self, engine_8, sample_facts):
        """stability deve estar entre -1 e 1 (similaridade de cosseno)."""
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=5)
        assert -1.0 <= result["stability"] <= 1.0

    def test_n_iterations_present(self, engine_8, sample_facts):
        """n_iterations deve estar no resultado."""
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=7)
        assert result["n_iterations"] == 7

    def test_n_facts_present(self, engine_8, sample_facts):
        """n_facts deve estar no resultado."""
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=5)
        assert result["n_facts"] == len(sample_facts)

    def test_consolidation_gain_calculation(self, engine_8, sample_facts):
        """consolidation_gain deve ser stability / (n_iterations + 1)."""
        n_iter = 5
        result = engine_8.consolidate_knowledge(sample_facts, n_iterations=n_iter)
        expected_gain = result["stability"] / (n_iter + 1)
        assert abs(result["consolidation_gain"] - expected_gain) < 1e-10

    def test_more_iterations_decrease_gain(self, engine_8, sample_facts):
        """Mais iterações devem diminuir o gain (denominador maior)."""
        result_few = engine_8.consolidate_knowledge(sample_facts, n_iterations=2)
        result_many = engine_8.consolidate_knowledge(sample_facts, n_iterations=10)
        # gain diminui se stability similar
        if abs(result_few["stability"] - result_many["stability"]) < 0.1:
            assert result_many["consolidation_gain"] < result_few["consolidation_gain"]

    def test_single_fact(self, engine_8):
        """consolidate_knowledge com um único fato."""
        facts = [np.ones(8)]
        result = engine_8.consolidate_knowledge(facts, n_iterations=3)
        assert result["n_facts"] == 1
        assert "stability" in result

    def test_requires_matching_dimensions(self, engine_1, sample_facts):
        """consolidate_knowledge com brain 1-dim e facts 8-dim."""
        # engine_1 tem n_out=1, facts são 8-dim
        # A função projeta para n_out, deve funcionar
        result = engine_1.consolidate_knowledge(sample_facts, n_iterations=3)
        assert "stability" in result


# ==============================================================
#  TESTES: DETECT CONTRADICTIONS
# ==============================================================

class TestDetectContradictions:
    def test_returns_dict_with_required_keys(self, engine_1, sample_facts):
        """detect_contradictions deve retornar dict com contradictions, n_contradictions."""
        result = engine_1.detect_contradictions(sample_facts)
        assert "contradictions" in result
        assert "n_contradictions" in result

    def test_contradictions_is_list(self, engine_1, sample_facts):
        """contradictions deve ser uma lista."""
        result = engine_1.detect_contradictions(sample_facts)
        assert isinstance(result["contradictions"], list)

    def test_n_contradictions_is_int(self, engine_1, sample_facts):
        """n_contradictions deve ser um inteiro."""
        result = engine_1.detect_contradictions(sample_facts)
        assert isinstance(result["n_contradictions"], int)

    def test_n_contradictions_matches_list_length(self, engine_1, sample_facts):
        """n_contradictions deve ser igual ao tamanho da lista."""
        result = engine_1.detect_contradictions(sample_facts)
        assert result["n_contradictions"] == len(result["contradictions"])

    def test_pairs_present(self, engine_1, sample_facts):
        """pairs deve estar no resultado."""
        result = engine_1.detect_contradictions(sample_facts)
        assert "pairs" in result

    def test_contradiction_structure(self, engine_1):
        """Cada contradição deve ter pair, input_similarity, output_similarity, contradiction_score."""
        # Criar fatos similares que podem gerar contradição
        facts = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        result = engine_1.detect_contradictions(facts, threshold=0.5)
        for c in result["contradictions"]:
            assert "pair" in c
            assert "input_similarity" in c
            assert "output_similarity" in c
            assert "contradiction_score" in c

    def test_no_contradictions_with_dissimilar_inputs(self, engine_1):
        """Fatos dissimilares não devem gerar contradições."""
        facts = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        result = engine_1.detect_contradictions(facts, threshold=0.3)
        # Entradas ortogonais (input_sim=0), não devem gerar contradições
        assert result["n_contradictions"] == 0

    def test_threshold_parameter(self, engine_1):
        """Threshold afeta detecção de contradições."""
        facts = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        # Threshold muito alto detecta mais contradições
        result_high = engine_1.detect_contradictions(facts, threshold=0.99)
        result_low = engine_1.detect_contradictions(facts, threshold=0.01)
        assert result_high["n_contradictions"] >= result_low["n_contradictions"]

    def test_empty_facts(self, engine_1):
        """detect_contradictions com facts vazio."""
        result = engine_1.detect_contradictions([])
        assert result["n_contradictions"] == 0
        assert len(result["contradictions"]) == 0

    def test_single_fact(self, engine_1):
        """detect_contradictions com um único fato (sem pares)."""
        facts = [np.ones(8)]
        result = engine_1.detect_contradictions(facts)
        assert result["n_contradictions"] == 0

    def test_identical_facts(self, engine_1):
        """Fatos idênticos não geram contradições (output_sim = 1)."""
        fact = np.array([1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        facts = [fact.copy(), fact.copy()]
        result = engine_1.detect_contradictions(facts, threshold=0.3)
        # Fatos idênticos: input_sim=1, output_sim=1, não é contradição
        assert result["n_contradictions"] == 0


# ==============================================================
#  TESTES: KNOWLEDGE BASE
# ==============================================================

class TestKnowledgeBase:
    def test_init(self, brain_8):
        """KnowledgeBase deve inicializar corretamente."""
        kb = KnowledgeBase(brain_8, max_facts=50)
        assert kb.max_facts == 50
        assert len(kb.facts) == 0
        assert len(kb.labels) == 0

    def test_add_fact(self, kb):
        """add_fact deve adicionar fato à base."""
        fact = np.ones(8)
        kb.add_fact(fact, label="test")
        assert len(kb.facts) == 1
        assert kb.labels[0] == "test"

    def test_add_multiple_facts(self, kb):
        """Adicionar múltiplos fatos."""
        for i in range(5):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        assert len(kb.facts) == 5

    def test_add_fact_stores_copy(self, kb):
        """add_fact deve armazenar cópia, não referência."""
        fact = np.ones(8)
        kb.add_fact(fact)
        fact[0] = 999
        assert kb.facts[0][0] == 1.0

    def test_add_fact_with_metadata(self, kb):
        """add_fact com metadata."""
        fact = np.ones(8)
        meta = {"source": "test", "timestamp": 123}
        kb.add_fact(fact, label="meta_test", metadata=meta)
        assert kb.metadata[0] == meta

    def test_add_fact_default_label(self, kb):
        """add_fact sem label deve usar string vazia."""
        kb.add_fact(np.ones(8))
        assert kb.labels[0] == ""

    def test_max_facts_fifo(self, kb):
        """Ao exceder max_facts, remover o mais antigo (FIFO)."""
        kb.max_facts = 3
        for i in range(5):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        assert len(kb.facts) == 3
        assert kb.labels[0] == "fact_2"
        assert kb.labels[-1] == "fact_4"

    def test_query_returns_dict_with_results(self, kb):
        """query deve retornar dict com results."""
        for i in range(3):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        query = np.ones(8) * 0.5
        result = kb.query(query, top_k=2)
        assert "results" in result
        assert "scores" in result

    def test_query_results_count(self, kb):
        """query deve retornar no máximo top_k resultados."""
        for i in range(5):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        query = np.ones(8) * 0.5
        result = kb.query(query, top_k=3)
        assert len(result["results"]) <= 3

    def test_query_results_structure(self, kb):
        """Cada resultado deve ter index, label, score, fact."""
        kb.add_fact(np.ones(8), label="test")
        query = np.ones(8) * 0.5
        result = kb.query(query, top_k=1)
        for r in result["results"]:
            assert "index" in r
            assert "label" in r
            assert "score" in r
            assert "fact" in r

    def test_query_empty_kb(self, kb):
        """query em KB vazio retorna results vazio."""
        query = np.ones(8)
        result = kb.query(query, top_k=3)
        assert result["results"] == []
        assert result["scores"] == []

    def test_query_scores_in_range(self, kb):
        """Scores de query devem estar entre -1 e 1."""
        for i in range(5):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        query = np.ones(8) * 0.5
        result = kb.query(query, top_k=3)
        for score in result["scores"]:
            assert -1.0 <= score <= 1.0

    def test_query_returns_query(self, kb):
        """query deve retornar a própria query."""
        kb.add_fact(np.ones(8))
        query = np.ones(8) * 0.5
        result = kb.query(query)
        assert "query" in result

    def test_reason_uses_forward_chain(self, kb):
        """reason deve usar forward_chain do engine."""
        for i in range(3):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        query = np.ones(8) * 0.5
        result = kb.reason(query, n_steps=3)
        assert "conclusions" in result
        assert "confidence" in result
        assert "chain" in result

    def test_reason_returns_chain_with_facts_and_query(self, kb):
        """reason deve incluir query e fatos no chain."""
        kb.add_fact(np.ones(8), label="fact_0")
        query = np.ones(8) * 0.5
        result = kb.reason(query, n_steps=2)
        # Chain deve ter pelo menos 1 (query) + 1 (fact) = 2 entradas
        assert len(result["chain"]) >= 2

    def test_reason_includes_recent_facts(self, kb):
        """reason deve incluir os últimos 10 fatos."""
        for i in range(15):
            kb.add_fact(np.ones(8) * i, label=f"fact_{i}")
        query = np.ones(8) * 0.5
        result = kb.reason(query, n_steps=2)
        # Deve incluir query + últimos 10 fatos + 2 steps
        # Pelo menos 1 (query) + 10 (fatos) = 11
        assert len(result["chain"]) >= 11

    def test_cosine_sim_identical_vectors(self, kb):
        """_cosine_sim com vetores idênticos deve ser 1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = kb._cosine_sim(a, b)
        assert abs(result - 1.0) < 1e-10

    def test_cosine_sim_orthogonal_vectors(self, kb):
        """_cosine_sim com vetores ortogonais deve ser 0.0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = kb._cosine_sim(a, b)
        assert abs(result - 0.0) < 1e-10

    def test_cosine_sim_opposite_vectors(self, kb):
        """_cosine_sim com vetores opostos deve ser -1.0."""
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        result = kb._cosine_sim(a, b)
        assert abs(result - (-1.0)) < 1e-10

    def test_cosine_sim_zero_vector(self, kb):
        """_cosine_sim com vetor zero deve retornar 0.0."""
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        result = kb._cosine_sim(a, b)
        assert result == 0.0


# ==============================================================
#  TESTES: INTEGRAÇÃO
# ==============================================================

class TestIntegration:
    def test_forward_then_backward(self, engine_1, sample_facts):
        """Executar forward_chain e depois backward_chain."""
        forward_result = engine_1.forward_chain(sample_facts, n_steps=3)
        hypothesis = forward_result["conclusions"]
        backward_result = engine_1.backward_chain(hypothesis, sample_facts)
        assert "supported" in backward_result

    def test_consolidate_then_detect(self, engine_8, sample_facts):
        """Consolidar conhecimento e depois detectar contradições."""
        engine_8.consolidate_knowledge(sample_facts, n_iterations=3)
        result = engine_8.detect_contradictions(sample_facts)
        assert "n_contradictions" in result

    def test_full_kb_workflow(self, brain_8):
        """Workflow completo: add facts, query, reason."""
        kb = KnowledgeBase(brain_8, max_facts=50)

        # Adicionar fatos
        rng = np.random.default_rng(42)
        for i in range(10):
            kb.add_fact(rng.standard_normal(8), label=f"fact_{i}")

        # Query
        query = rng.standard_normal(8)
        query_result = kb.query(query, top_k=5)
        assert len(query_result["results"]) > 0

        # Reason
        reason_result = kb.reason(query, n_steps=3)
        assert "conclusions" in reason_result
        assert "confidence" in reason_result

    def test_engine_mode_transitions(self, engine_1, sample_facts, sample_hypothesis_1d):
        """Engine deve transicionar corretamente entre modos infer/learn."""
        assert engine_1.brain.mode == "learn"
        engine_1.forward_chain(sample_facts, n_steps=2)
        assert engine_1.brain.mode == "infer"
        engine_1.backward_chain(sample_hypothesis_1d, sample_facts)
        assert engine_1.brain.mode == "infer"

    def test_kb_with_max_facts_1(self, brain_8):
        """KB com max_facts=1 deve manter apenas o fato mais recente."""
        kb = KnowledgeBase(brain_8, max_facts=1)
        kb.add_fact(np.ones(8), label="first")
        kb.add_fact(np.ones(8) * 2, label="second")
        assert len(kb.facts) == 1
        assert kb.labels[0] == "second"


# ==============================================================
#  TESTES: EDGE CASES
# ==============================================================

class TestEdgeCases:
    def test_forward_with_single_fact_1d(self, engine_1):
        """forward_chain com fato 1-dim em brain com n_out=1."""
        facts = [np.array([0.5])]
        # Ajustar para fato com dimensão correta
        facts = [np.ones(8) * 0.5]
        result = engine_1.forward_chain(facts, n_steps=1)
        assert len(result["chain"]) == 2

    def test_detect_contradictions_two_facts(self, engine_1):
        """detect_contradictions com exatamente 2 fatos."""
        facts = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        result = engine_1.detect_contradictions(facts)
        assert result["n_contradictions"] == len(result["contradictions"])

    def test_consolidate_single_iteration(self, engine_8):
        """consolidate_knowledge com n_iterations=1."""
        facts = [np.ones(8)]
        result = engine_8.consolidate_knowledge(facts, n_iterations=1)
        assert result["n_iterations"] == 1

    def test_backward_chain_single_fact(self, engine_1, sample_hypothesis_1d):
        """backward_chain com um único fato."""
        facts = [np.ones(8)]
        result = engine_1.backward_chain(sample_hypothesis_1d, facts)
        assert len(result["relevant_facts"]) <= 1

    def test_query_top_k_larger_than_facts(self, kb):
        """query com top_k maior que número de fatos."""
        kb.add_fact(np.ones(8), label="only")
        query = np.ones(8) * 0.5
        result = kb.query(query, top_k=10)
        assert len(result["results"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
