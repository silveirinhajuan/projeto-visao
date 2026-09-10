#!/usr/bin/env python3
"""causal.py — Dynamic Causal Model (DCM) properties of liquid neural networks.

Baseado em: "Robust flight navigation out of distribution with liquid neural networks"
(Chahine et al., Science Robotics 2023) e "Liquid neural networks as dynamic causal models"
(Vorbach et al., 2021).

O paper prova que LTCs (Liquid Time-Constant networks) são DCMs — modelos causais
dinâmicos. Isso significa que:

1. Capturam causalidade (não apenas correlação)
2. São robustos a intervenções externas e internas
3. Generalizam OOD porque aprendem a CAUSA da tarefa, não o contexto

Este módulo implementa ferramentas para analisar as propriedades causais do VISÃO.
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional, Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class CausalAnalyzer:
    """Analisa as propriedades causais de uma rede líquida.
    
    Baseado na metodologia de Chahine et al. (2023):
    - Intervenção: modificar entrada e medir efeito na saída
    - Contrafactual: "o que acontecia se a entrada fosse diferente?"
    - Causal attribution: quais entradas causam quais saídas
    """
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
    
    def intervene(self, x: np.ndarray, intervention: dict) -> dict:
        """Aplica uma intervenção causal na entrada.
        
        Parameters
        ----------
        x : array (n_in,)
            Entrada original.
        intervention : dict
            {índice: valor} — fixa entrada[i] = valor.
            
        Returns
        -------
        dict com 'original_output', 'intervened_output', 'causal_effect'
        """
        original = self.brain.forward(x.copy())
        
        x_intervened = x.copy()
        for idx, val in intervention.items():
            x_intervened[idx] = val
        
        intervened = self.brain.forward(x_intervened)
        
        causal_effect = float(np.mean((intervened - original) ** 2))
        
        return {
            'original_output': original,
            'intervened_output': intervened,
            'causal_effect': causal_effect,
            'intervention': intervention,
        }
    
    def counterfactual(self, x: np.ndarray, 
                       original_intervention: dict,
                       counterfactual_intervention: dict) -> dict:
        """Gera contrafactual: compara duas intervenções diferentes.
        
        "O que acontecia se em vez de X, fosse Y?"
        """
        result_original = self.intervene(x, original_intervention)
        result_counter = self.intervene(x, counterfactual_intervention)
        
        return {
            'original': result_original,
            'counterfactual': result_counter,
            'counterfactual_effect': float(np.mean(
                (result_counter['intervened_output'] - result_original['intervened_output']) ** 2
            )),
        }
    
    def causal_attribution(self, x: np.ndarray, 
                           target_idx: int = 0,
                           method: str = 'gradient') -> np.ndarray:
        """Atribui causalidade a cada entrada para uma saída específica.
        
        Parameters
        ----------
        x : array (n_in,)
        target_idx : índice da saída de interesse
        method : 'gradient' ou 'intervention'
        
        Returns
        -------
        array (n_in,) com scores de causalidade
        """
        if method == 'gradient':
            return self._gradient_attribution(x, target_idx)
        elif method == 'intervention':
            return self._intervention_attribution(x, target_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _gradient_attribution(self, x: np.ndarray, target_idx: int) -> np.ndarray:
        """Atribuição via gradiente (aproximação rápida)."""
        eps = 1e-4
        base = self.brain.forward(x)
        attribution = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            out_plus = self.brain.forward(x_plus)
            attribution[i] = (out_plus[target_idx] - base[target_idx]) / eps
        
        return attribution
    
    def _intervention_attribution(self, x: np.ndarray, target_idx: int) -> np.ndarray:
        """Atribuição via intervenção (mais precisa, mais lenta)."""
        base = self.brain.forward(x)
        attribution = np.zeros_like(x)
        
        for i in range(len(x)):
            # Fixar entrada[i] para 0 (intervenção)
            x_intervened = x.copy()
            x_intervened[i] = 0.0
            out_intervened = self.brain.forward(x_intervened)
            attribution[i] = base[target_idx] - out_intervened[target_idx]
        
        return attribution
    
    def compute_causal_strength(self, x: np.ndarray) -> dict:
        """Mede a força causal de cada entrada.
        
        Returns dict com métricas de causalidade.
        """
        n_in = len(x)
        gradient_attr = self._gradient_attribution(x, 0)
        intervention_attr = self._intervention_attribution(x, 0)
        
        # Correlação entre gradiente e intervenção (devem concordar em LNNs)
        correlation = np.corrcoef(gradient_attr, intervention_attr)[0, 1]
        
        return {
            'gradient_attribution': gradient_attr,
            'intervention_attribution': intervention_attr,
            'attribution_correlation': float(correlation),
            'top_causal_inputs': np.argsort(np.abs(gradient_attr))[-3:].tolist(),
            'causal_sparsity': float(np.sum(np.abs(gradient_attr) > 0.01) / n_in),
        }
    
    def ood_robustness_predictor(self, x: np.ndarray) -> float:
        """Prevê robustez OOD baseado em propriedades causais.
        
        Hipótese do paper: redes que capturam causalidade (atribuição esparse
        e focalizada) generalizam melhor OOD.
        
        Returns score [0, 1] onde 1 = muito robusto.
        """
        causal = self.compute_causal_strength(x)
        
        # Atribuição esparse = focada na tarefa = robusto
        sparsity = 1.0 - causal['causal_sparsity']
        
        # Alta correlação gradiente-intervenção = causal
        causality = abs(causal['attribution_correlation'])
        
        return float(0.5 * sparsity + 0.5 * causality)


class CausalGeneralizationTest:
    """Teste de generalização baseado em causalidade.
    
    Metodologia do paper:
    1. Treinar em ambiente A
    2. Intervenir em características não-causais (ex: fundo)
    3. Medir se a saída muda (não deve mudar se aprendeu causalidade)
    """
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.analyzer = CausalAnalyzer(brain)
    
    def test_context_invariance(self, 
                                 x: np.ndarray,
                                 causal_indices: list,
                                 n_perturbations: int = 10) -> dict:
        """Testa se a saída é invariante a mudanças no contexto (não-causal).
        
        Parameters
        ----------
        x : entrada base
        causal_indices : índices das entradas causais (não perturbar)
        n_perturbations : número de perturbações
        
        Returns
        -------
        dict com métricas de invariância
        """
        base_output = self.brain.forward(x)
        n_in = len(x)
        non_causal_indices = [i for i in range(n_in) if i not in causal_indices]
        
        if not non_causal_indices:
            return {'invariance_score': 1.0, 'mean_deviation': 0.0}
        
        deviations = []
        for _ in range(n_perturbations):
            x_perturbed = x.copy()
            # Perturbar entradas não-causais
            for idx in non_causal_indices:
                x_perturbed[idx] += np.random.normal(0, 0.5)
            
            perturbed_output = self.brain.forward(x_perturbed)
            deviation = float(np.mean((perturbed_output - base_output) ** 2))
            deviations.append(deviation)
        
        mean_deviation = np.mean(deviations)
        invariance = 1.0 / (1.0 + mean_deviation)
        
        return {
            'invariance_score': float(invariance),
            'mean_deviation': float(mean_deviation),
            'std_deviation': float(np.std(deviations)),
            'n_perturbations': n_perturbations,
        }
    
    def test_causal_intervention(self,
                                  x: np.ndarray,
                                  causal_indices: list,
                                  intervention_value: float = 0.0) -> dict:
        """Testa se intervenção em entradas causais muda a saída."""
        base_output = self.brain.forward(x)
        
        x_intervened = x.copy()
        for idx in causal_indices:
            x_intervened[idx] = intervention_value
        
        intervened_output = self.brain.forward(x_intervened)
        causal_effect = float(np.mean((intervened_output - base_output) ** 2))
        
        return {
            'causal_effect': causal_effect,
            'output_changed': causal_effect > 0.01,
            'intervention_magnitude': float(np.sqrt(np.mean((x_intervened - x) ** 2))),
        }


if __name__ == "__main__":
    print("="*60)
    print("CAUSAL ANALYSIS MODULE DEMO")
    print("Based on Chahine et al. (2023), Science Robotics")
    print("="*60)
    
    brain = VisaoBrain(8, 32, 1, seed=42)
    brain.set_mode('infer')
    analyzer = CausalAnalyzer(brain)
    
    x = np.random.randn(8)
    
    # Intervention
    print("\n[1] Causal Intervention")
    result = analyzer.intervene(x, {0: 0.0, 1: 0.0})
    print(f"    Causal effect: {result['causal_effect']:.4f}")
    
    # Attribution
    print("\n[2] Causal Attribution")
    attr = analyzer.causal_attribution(x, method='gradient')
    print(f"    Top causal inputs: {np.argsort(np.abs(attr))[-3:]}")
    
    # Causal strength
    print("\n[3] Causal Strength")
    strength = analyzer.compute_causal_strength(x)
    print(f"    Attribution correlation: {strength['attribution_correlation']:.4f}")
    print(f"    Causal sparsity: {strength['causal_sparsity']:.4f}")
    
    # OOD prediction
    print("\n[4] OOD Robustness Prediction")
    robustness = analyzer.ood_robustness_predictor(x)
    print(f"    Predicted robustness: {robustness:.4f}")
    
    print("\n" + "="*60)
    print("CAUSAL ANALYSIS OK!")
    print("="*60)
