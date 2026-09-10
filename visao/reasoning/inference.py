#!/usr/bin/env python3
"""inference.py — Módulo de inferência e querying do VisaoCognitiveBrain.

Permite ao cérebro cognitivo:
1. Responder perguntas sobre o que aprendeu
2. Fazer inferências lógicas baseadas no conhecimento armazenado
3. Consolidar conhecimento via raciocínio repetido
4. Detectar contradições no conhecimento

Baseado em: Neural Theorem Provers (Rocktäschel & Riedel, 2017) +
Neural Program Interpreters (Pierrot et al., 2021).
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class InferenceEngine:
    """Motor de inferência para o VisaoCognitiveBrain.
    
    Usa o estado interno do cérebro para fazer inferências:
    - Forward chaining: a partir de fatos, derivar conclusões
    - Backward chaining: a partir de hipóteses, verificar suporte
    - Consolidação: reforçar conhecimento via raciocínio repetido
    """
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
    
    def forward_chain(self, facts: list, n_steps: int = 5) -> dict:
        """Forward chaining: deriva conclusões a partir de fatos.
        
        Parameters
        ----------
        facts : list of np.ndarray
            Vetores de fatos conhecidos
        n_steps : int
            Número de passos de inferência
            
        Returns
        -------
        dict com 'conclusions', 'confidence', 'chain'
        """
        self.brain.set_mode('infer')
        self.brain.reset_state()
        
        chain = []
        current_state = None
        
        # Processar cada fato
        for i, fact in enumerate(facts):
            output = self.brain.forward(fact)
            chain.append({
                'step': i,
                'input': fact,
                'output': output,
                'type': 'fact'
            })
            current_state = output
        
        # Passos de inferência (raciocínio)
        for step in range(n_steps):
            if current_state is None:
                break
            # Projetar para dimensao de entrada para chaining
            if len(current_state) < len(facts[0]):
                padded = np.zeros(len(facts[0]))
                padded[:len(current_state)] = current_state
            else:
                padded = current_state[:len(facts[0])]
            next_output = self.brain.forward(padded)
            chain.append({
                'step': len(facts) + step,
                'input': padded,
                'output': next_output,
                'type': 'inference'
            })
            current_state = next_output
        
        # Confiança baseada na estabilidade da cadeia
        confidence = self._compute_confidence(chain)
        
        return {
            'conclusions': current_state,
            'confidence': confidence,
            'chain': chain,
            'n_facts': len(facts),
            'n_inference_steps': n_steps,
        }
    
    def backward_chain(self, hypothesis: np.ndarray, facts: list, n_steps: int = 5) -> dict:
        """Backward chaining: verifica se fatos suportam uma hipótese.
        
        Parameters
        ----------
        hypothesis : np.ndarray
            Hipótese a ser verificada
        facts : list of np.ndarray
            Fatos disponíveis
        n_steps : int
            Número de passos de verificação
            
        Returns
        -------
        dict com 'supported', 'support_score', 'relevant_facts'
        """
        self.brain.set_mode('infer')
        
        # Calcular similaridade da hipótese com cada fato
        fact_scores = []
        for fact in facts:
            self.brain.reset_state()
            fact_output = self.brain.forward(fact)
            # Similaridade entre fato e hipótese
            sim = self._cosine_sim(fact_output.flatten(), hypothesis.flatten())
            fact_scores.append(float(sim))
        
        # Fatos mais relevantes
        relevant_indices = np.argsort(fact_scores)[-3:][::-1]
        relevant_facts = [facts[i] for i in relevant_indices if i < len(facts)]
        support_score = np.mean(fact_scores) if fact_scores else 0.0
        
        return {
            'supported': support_score > 0.5,
            'support_score': float(support_score),
            'relevant_facts': relevant_facts,
            'fact_scores': fact_scores,
        }
    
    def consolidate_knowledge(self, facts: list, n_iterations: int = 10) -> dict:
        """Consolida conhecimento via raciocínio repetido.
        
        Baseado em: a repetição espaçada fortalece conexões neurais.
        O cérebro processa os fatos múltiplas vezes para consolidar.
        
        Parameters
        ----------
        facts : list of np.ndarray
            Fatos a consolidar
        n_iterations : int
            Número de iterações de consolidação
            
        Returns
        -------
        dict com 'stability', 'consolidation_gain'
        """
        self.brain.set_mode('infer')
        
        initial_outputs = []
        final_outputs = []
        
        # Primeira passagem (antes da consolidação)
        for fact in facts:
            self.brain.reset_state()
            output = self.brain.forward(fact)
            initial_outputs.append(output.copy())
        
        # Consolidação: repetir processamento múltiplas vezes
        self.brain.set_mode('learn')
        n_out = self.brain.n_out
        for iteration in range(n_iterations):
            for fact in facts:
                self.brain.reset_state()
                # Projetar fact para dimensao de saida (auto-associativo)
                if len(fact) >= n_out:
                    target = fact[:n_out]
                else:
                    target = np.zeros(n_out)
                    target[:len(fact)] = fact
                self.brain.learn(fact, target)
        
        # Segunda passagem (depois da consolidação)
        self.brain.set_mode('infer')
        for fact in facts:
            self.brain.reset_state()
            output = self.brain.forward(fact)
            final_outputs.append(output.copy())
        
        # Medir estabilidade (quanto a saída mudou)
        stability = np.mean([
            self._cosine_sim(init.flatten(), fin.flatten())
            for init, fin in zip(initial_outputs, final_outputs)
        ])
        
        # Ganho de consolidação (deveria aumentar a estabilidade)
        consolidation_gain = stability / (n_iterations + 1)
        
        return {
            'stability': float(stability),
            'consolidation_gain': float(consolidation_gain),
            'n_iterations': n_iterations,
            'n_facts': len(facts),
        }
    
    def detect_contradictions(self, facts: list, threshold: float = 0.3) -> dict:
        """Detecta contradições no conhecimento.
        
        Uma contradição ocorre quando dois fatos similares produzem
        saídas muito diferentes (ou vice-versa).
        
        Parameters
        ----------
        facts : list of np.ndarray
            Fatos a verificar
        threshold : float
            Limiar de similaridade para considerar contradição
            
        Returns
        -------
        dict com 'contradictions', 'n_contradictions', 'pairs'
        """
        self.brain.set_mode('infer')
        
        outputs = []
        for fact in facts:
            self.brain.reset_state()
            output = self.brain.forward(fact)
            outputs.append(output)
        
        contradictions = []
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                # Similaridade de entrada
                input_sim = self._cosine_sim(facts[i].flatten(), facts[j].flatten())
                # Similaridade de saída
                output_sim = self._cosine_sim(outputs[i].flatten(), outputs[j].flatten())
                
                # Contradição: entrada similar mas saída muito diferente
                if input_sim > 0.8 and output_sim < threshold:
                    contradictions.append({
                        'pair': (i, j),
                        'input_similarity': float(input_sim),
                        'output_similarity': float(output_sim),
                        'contradiction_score': float(input_sim - output_sim),
                    })
        
        return {
            'contradictions': contradictions,
            'n_contradictions': len(contradictions),
            'pairs': [(c['pair'][0], c['pair'][1]) for c in contradictions],
        }
    
    def _compute_confidence(self, chain: list) -> float:
        """Calcula confiança baseada na estabilidade da cadeia de inferência."""
        if len(chain) < 2:
            return 1.0
        
        # Confiança = similaridade média entre passos consecutuous
        similarities = []
        for i in range(1, len(chain)):
            prev_out = chain[i-1]['output'].flatten()
            curr_out = chain[i]['output'].flatten()
            sim = self._cosine_sim(prev_out, curr_out)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade de cosseno entre dois vetores."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class KnowledgeBase:
    """Base de conhecimento para o VisaoCognitiveBrain.
    
    Armazena fatos e permite inferências sobre eles.
    Usa o InferenceEngine para processar.
    """
    
    def __init__(self, brain: VisaoBrain, max_facts: int = 1000):
        self.brain = brain
        self.engine = InferenceEngine(brain)
        self.max_facts = max_facts
        self.facts: list[np.ndarray] = []
        self.labels: list[str] = []
        self.metadata: list[dict] = []
    
    def add_fact(self, fact: np.ndarray, label: str = "", metadata: dict = None):
        """Adiciona um fato à base de conhecimento."""
        if len(self.facts) >= self.max_facts:
            # Remover fato mais antigo (FIFO)
            self.facts.pop(0)
            self.labels.pop(0)
            self.metadata.pop(0)
        
        self.facts.append(fact.copy())
        self.labels.append(label)
        self.metadata.append(metadata or {})
    
    def query(self, query: np.ndarray, top_k: int = 5) -> dict:
        """Consulta a base de conhecimento.
        
        Encontra os fatos mais relevantes para a query.
        """
        if not self.facts:
            return {'results': [], 'scores': []}
        
        scores = []
        for fact in self.facts:
            self.brain.set_mode('infer')
            self.brain.reset_state()
            fact_output = self.brain.forward(fact)
            query_output = self.brain.forward(query)
            sim = self._cosine_sim(fact_output.flatten(), query_output.flatten())
            scores.append(float(sim))
        
        # Top-k
        indices = np.argsort(scores)[-top_k:][::-1]
        results = [{
            'index': int(i),
            'label': self.labels[i],
            'score': scores[i],
            'fact': self.facts[i],
        } for i in indices]
        
        return {
            'results': results,
            'scores': [scores[i] for i in indices],
            'query': query,
        }
    
    def reason(self, query: np.ndarray, n_steps: int = 5) -> dict:
        """Raciocina sobre uma query usando forward chaining."""
        return self.engine.forward_chain([query] + self.facts[-10:], n_steps=n_steps)
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade de cosseno."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":
    print("="*60)
    print("INFERENCE ENGINE DEMO")
    print("="*60)
    
    brain = VisaoBrain(8, 32, 1, seed=42)
    engine = InferenceEngine(brain)
    
    # Forward chaining
    print("\n[1] Forward Chaining")
    facts = [np.random.randn(8) for _ in range(3)]
    result = engine.forward_chain(facts, n_steps=5)
    print(f"    Confidence: {result['confidence']:.4f}")
    print(f"    Chain length: {len(result['chain'])}")
    
    # Backward chaining
    print("\n[2] Backward Chaining")
    hypothesis = np.random.randn(1)
    result = engine.backward_chain(hypothesis, facts)
    print(f"    Supported: {result['supported']}")
    print(f"    Support score: {result['support_score']:.4f}")
    
    # Knowledge consolidation (usar brain com n_in == n_out para auto-associativo)
    print("\n[3] Knowledge Consolidation")
    brain_8 = VisaoBrain(8, 32, 8, seed=42)  # n_in == n_out
    engine_8 = InferenceEngine(brain_8)
    facts_8 = [np.random.randn(8) for _ in range(3)]
    result = engine_8.consolidate_knowledge(facts_8, n_iterations=5)
    print(f"    Stability: {result['stability']:.4f}")
    print(f"    Consolidation gain: {result['consolidation_gain']:.4f}")
    
    # Contradiction detection
    print("\n[4] Contradiction Detection")
    # Criar fatos contraditórios
    contradictory_facts = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        np.array([1, 0, 0, 0, 0, 0, 0, 0.1], dtype=float),  # Similar input
    ]
    result = engine.detect_contradictions(contradictory_facts, threshold=0.5)
    print(f"    Contradictions found: {result['n_contradictions']}")
    
    # Knowledge Base
    print("\n[5] Knowledge Base")
    kb = KnowledgeBase(brain, max_facts=100)
    for i in range(5):
        kb.add_fact(np.random.randn(8), label=f"fact_{i}")
    query = np.random.randn(8)
    result = kb.query(query, top_k=3)
    print(f"    Facts stored: {len(kb.facts)}")
    print(f"    Query results: {len(result['results'])}")
    
    print("\n" + "="*60)
    print("INFERENCE ENGINE OK!")
    print("="*60)
