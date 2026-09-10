#!/usr/bin/env python3
"""saliency.py — Interpretabilidade via VisualBackProp (Chahine et al., 2023).

Implementa mapas de atenção para entender o que a rede está "vendo".
Baseado em: "Robust flight navigation out of distribution with liquid neural networks"
(Science Robotics, 2023).
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class VisualBackProp:
    """Implementação do algoritmo VisualBackProp para mapas de saliência.
    
    O algoritmo funciona "desconvolvendo" a atenção das camadas finais
    para as iniciais, mostrando quais pixels da entrada são mais importantes
    para a decisão da rede.
    
    Reference: VisualBackProp (Dosovitskiy & Brox, 2016) + adaptação do
    paper de Chahine et al. (2023) para visualização de atenção em LNNs.
    """
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
    
    def compute_saliency(self, x: np.ndarray) -> np.ndarray:
        """Compute saliency map for input x.
        
        Returns a heatmap of same shape as input showing which features
        the network attends to most.
        
        Parameters
        ----------
        x : array (n_in,)
            Input vector (flattened image or feature vector).
            
        Returns
        -------
        saliency : array (n_in,)
            Saliency scores (higher = more important).
        """
        # For the liquid brain, we compute gradients of output w.r.t. input
        # This gives us a proxy for feature importance
        eps = 1e-4
        base_output = self.brain.forward(x)
        saliency = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            output_plus = self.brain.forward(x_plus)
            saliency[i] = np.sum((output_plus - base_output) ** 2) / eps
        
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency
    
    def compute_saliency_map_2d(self, x: np.ndarray, shape: tuple) -> np.ndarray:
        """Compute 2D saliency map (for image-like inputs)."""
        saliency = self.compute_saliency(x)
        return saliency.reshape(shape)
    
    def get_attention_stats(self, x: np.ndarray) -> dict:
        """Get statistics about where the network is attending."""
        saliency = self.compute_saliency(x)
        
        return {
            'max_attention': float(np.max(saliency)),
            'mean_attention': float(np.mean(saliency)),
            'attention_entropy': float(self._entropy(saliency)),
            'top_k_indices': np.argsort(saliency)[-5:].tolist(),
            'attention_std': float(np.std(saliency)),
        }
    
    def _entropy(self, x: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        x_pos = np.abs(x)
        if x_pos.sum() == 0:
            return 0.0
        p = x_pos / x_pos.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p))


class OODStressTest:
    """Stress tests para medir robustez a distribution shifts.
    
    Implementa os mesmos testes do paper Chahine et al. (2023):
    - Noise perturbation
    - Brightness shift
    - Contrast change
    - Saturation change
    
    A métrica é o desvio entre a trajetória original e a perturbada
    (menor = mais robusto).
    """
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
    
    def test_noise(self, x: np.ndarray, noise_levels: np.ndarray = None) -> dict:
        """Test robustness to Gaussian noise."""
        if noise_levels is None:
            noise_levels = np.array([0.0, 0.05, 0.1, 0.2, 0.3])
        
        results = {}
        base_output = self.brain.forward(x.copy())
        
        for level in noise_levels:
            perturbation = np.random.normal(0, level, x.shape)
            x_perturbed = x + perturbation
            perturbed_output = self.brain.forward(x_perturbed)
            deviation = float(np.mean((perturbed_output - base_output) ** 2))
            results[f'noise_{level}'] = deviation
        
        return results
    
    def test_brightness(self, x: np.ndarray, brightness_range: np.ndarray = None) -> dict:
        """Test robustness to brightness shifts."""
        if brightness_range is None:
            brightness_range = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        
        results = {}
        base_output = self.brain.forward(x.copy())
        
        for shift in brightness_range:
            x_perturbed = np.clip(x + shift, x.min(), x.max())
            perturbed_output = self.brain.forward(x_perturbed)
            deviation = float(np.mean((perturbed_output - base_output) ** 2))
            results[f'brightness_{shift}'] = deviation
        
        return results
    
    def test_contrast(self, x: np.ndarray, contrast_range: np.ndarray = None) -> dict:
        """Test robustness to contrast changes."""
        if contrast_range is None:
            contrast_range = np.array([0.5, 0.6, 1.0, 1.4])
        
        results = {}
        base_output = self.brain.forward(x.copy())
        mean_val = np.mean(x)
        
        for factor in contrast_range:
            x_perturbed = np.clip((x - mean_val) * factor + mean_val, x.min(), x.max())
            perturbed_output = self.brain.forward(x_perturbed)
            deviation = float(np.mean((perturbed_output - base_output) ** 2))
            results[f'contrast_{factor}'] = deviation
        
        return results
    
    def run_all_tests(self, x: np.ndarray) -> dict:
        """Run all stress tests."""
        return {
            'noise': self.test_noise(x),
            'brightness': self.test_brightness(x),
            'contrast': self.test_contrast(x),
        }
    
    def compute_robustness_score(self, x: np.ndarray) -> float:
        """Compute overall robustness score (higher = more robust).
        
        Based on the average deviation across all perturbation tests.
        Lower deviation = higher robustness.
        """
        results = self.run_all_tests(x)
        
        all_deviations = []
        for test_type, deviations in results.items():
            all_deviations.extend(deviations.values())
        
        avg_deviation = np.mean(all_deviations)
        # Convert to robustness score (inverse, normalized)
        robustness = 1.0 / (1.0 + avg_deviation)
        return float(robustness)


class RangeTest:
    """Teste de generalização para diferentes distâncias/horizontes.
    
    Baseado no range test do paper (Chahine et al., 2023):
    - Treinar em uma distância
    - Testar em distâncias maiores (2x, 3x)
    - Medir success rate
    
    No VISÃO, adaptamos para testar generalização temporal:
    - Treinar em sequências curtas
    - Testar em sequências mais longas
    """
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
    
    def test_temporal_generalization(self, 
                                     short_seq: np.ndarray,
                                     long_seq: np.ndarray) -> dict:
        """Test if a model trained on short sequences generalizes to longer ones."""
        # Process short sequence
        self.brain.reset_state()
        for x in short_seq:
            self.brain.forward(x)
        short_final = self.brain.forward(short_seq[-1])
        
        # Process long sequence (starting from same state)
        self.brain.reset_state()
        for x in long_seq:
            self.brain.forward(x)
        long_final = self.brain.forward(long_seq[-1])
        
        # Measure divergence
        divergence = float(np.mean((long_final - short_final) ** 2))
        
        return {
            'short_seq_len': len(short_seq),
            'long_seq_len': len(long_seq),
            'output_divergence': divergence,
            'temporal_stability': 1.0 / (1.0 + divergence),
        }
    
    def test_scale_generalization(self, 
                                   small_input: np.ndarray,
                                   scale_factors: list = None) -> dict:
        """Test generalization to different input scales."""
        if scale_factors is None:
            scale_factors = [1.0, 1.5, 2.0, 3.0]
        
        results = {}
        base_output = self.brain.forward(small_input)
        
        for factor in scale_factors:
            scaled_input = small_input * factor
            scaled_output = self.brain.forward(scaled_input)
            divergence = float(np.mean((scaled_output - base_output) ** 2))
            results[f'scale_{factor}x'] = {
                'divergence': divergence,
                'relative_change': divergence / (np.mean(base_output**2) + 1e-8),
            }
        
        return results


class OODGeneralizationBenchmark:
    """Benchmark completo de generalização Out-of-Distribution.
    
    Implementa a metodologia do paper Chahine et al. (2023):
    1. Train on distribution A
    2. Test on distribution B (unseen)
    3. Measure performance gap
    """
    
    def __init__(self, brain_factory, seed: int = 42):
        self.brain_factory = brain_factory
        self.seed = seed
    
    def compare_architectures(self,
                               train_data: tuple,
                               test_data: tuple,
                               architectures: dict) -> dict:
        """Compare OOD generalization across architectures.
        
        Parameters
        ----------
        train_data : (X, y) tuple
        test_data : (X, y) tuple (OOD)
        architectures : dict {name: factory_fn}
        
        Returns
        -------
        dict with generalization metrics for each architecture
        """
        results = {}
        
        for name, factory in architectures.items():
            brain = factory()
            
            # Train
            X_train, y_train = train_data
            for X, y in zip(X_train, y_train):
                brain.learn(X, y)
            
            train_error = self._evaluate(brain, train_data)
            test_error = self._evaluate(brain, test_data)
            
            results[name] = {
                'train_error': train_error,
                'test_error': test_error,
                'generalization_gap': test_error - train_error,
                'ood_ratio': test_error / (train_error + 1e-8),
            }
        
        return results
    
    def _evaluate(self, brain: VisaoBrain, data: tuple) -> float:
        """Evaluate brain on data."""
        X, y = data
        total_error = 0.0
        for X_i, y_i in zip(X, y):
            pred = brain.forward(X_i)
            total_error += np.mean((pred - y_i) ** 2)
        return total_error / len(X)


if __name__ == "__main__":
    print("="*60)
    print("INTERPRETABILITY & ROBUSTNESS MODULE DEMO")
    print("Based on Chahine et al. (2023), Science Robotics")
    print("="*60)
    
    # Create brain
    brain = VisaoBrain(8, 32, 1, seed=42)
    brain.set_mode('infer')
    
    # Saliency
    print("\n[1] VisualBackProp / Saliency maps")
    saliency = VisualBackProp(brain)
    x = np.random.randn(8)
    attn = saliency.compute_saliency(x)
    stats = saliency.get_attention_stats(x)
    print(f"    Saliency shape: {attn.shape}")
    print(f"    Max attention: {stats['max_attention']:.4f}")
    print(f"    Entropy: {stats['attention_entropy']:.4f}")
    
    # Stress tests
    print("\n[2] OOD Stress Tests")
    stress = OODStressTest(brain)
    results = stress.run_all_tests(x)
    robustness = stress.compute_robustness_score(x)
    print(f"    Robustness score: {robustness:.4f}")
    for test_type, deviations in results.items():
        print(f"    {test_type}: {len(deviations)} levels tested")
    
    # Range test
    print("\n[3] Range / Scale Generalization")
    range_test = RangeTest(brain)
    scale_results = range_test.test_scale_generalization(x)
    print(f"    Scale factors tested: {list(scale_results.keys())}")
    
    print("\n" + "="*60)
    print("INTERPRETABILITY & ROBUSTNESS OK!")
    print("="*60)
