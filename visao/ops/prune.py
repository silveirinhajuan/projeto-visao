"""
prune.py — Tarefa 11.0: Poda de sinapses inúteis.

Princípio: sinapse com omega baixo por muito tempo é inútil.

Funcionalidades:
  - Identifica sinapses com importância (omega) abaixo de um limiar
  - Remove sinapses (zera permanentemente)
  - Mantém sparsity da rede abaixo de um target (ex: 80%)
  - Compatível com VisaoBrain

Mecanismos de importância:
  - W_out (readout): usa omega diretamente (rastreado pelo learner)
  - W_rec (recorrente): usa |W_rec| como proxy de importância (não há omega por sinapse)

Uso:
    from visao.ops.prune import SynapsePruner, compute_sparsity

    pruner = SynapsePruner(threshold=0.01, target_sparsity=0.80)
    stats = pruner.prune(brain)
    print(stats)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ==============================================================
#  MÉTRICAS DE SPARSITY
# ==============================================================

def compute_sparsity(brain) -> dict:
    """Calcula sparsity de cada camada e total do brain.

    Retorna dict com:
      - total_sparsity: fração de zeros em todos os pesos estruturais
      - rec_sparsity: sparsity do W_rec (reservatório)
      - out_sparsity: sparsity do W_out (readout)
      - total_params: número total de parâmetros estruturais
      - total_zeros: número total de sinapses zeradas
    """
    cell = brain.cell
    learner = brain.learner

    # W_rec: conta zeros (incluindo máscara estrutural)
    rec_total = cell.W_rec.size
    rec_zeros = int(np.sum(cell.W_rec == 0))

    # W_out: conta zeros
    out_total = learner.W_out.size
    out_zeros = int(np.sum(learner.W_out == 0))

    total = rec_total + out_total
    total_zeros = rec_zeros + out_zeros

    return {
        "total_sparsity": total_zeros / total if total > 0 else 0.0,
        "rec_sparsity": rec_zeros / rec_total if rec_total > 0 else 0.0,
        "out_sparsity": out_zeros / out_total if out_total > 0 else 0.0,
        "total_params": total,
        "total_zeros": total_zeros,
        "rec_total": rec_total,
        "rec_zeros": rec_zeros,
        "out_total": out_total,
        "out_zeros": out_zeros,
    }


# ==============================================================
#  PODADOR DE SINAPSES
# ==============================================================

class SynapsePruner:
    """Poda sinapses inúteis baseada em importância (omega).

    Estratégia:
      1. Computa importância de cada sinapse:
         - W_out: usa omega (importância EWC rastreada)
         - W_rec: usa |W_rec| como proxy (não há omega por sinapse)
      2. Identifica sinapses abaixo do limiar (threshold)
      3. Ordena por importância crescente (menos importante primeiro)
      4. Podar apenas o necessário para atingir target_sparsity
         (nunca poda além do target)
      5. Zera permanentemente os pesos podados

    Parâmetros
    ----------
    threshold : float
        Limiar de importância. Sinapses com importância abaixo disso são
        candidatas a poda.
    target_sparsity : float
        Sparsity máxima desejada (ex: 0.80 = 80% de zeros).
        O pruner nunca ultrapassa esse limite.
    prune_rec : bool
        Se True, poda também W_rec (usando |W_rec| como importância).
        Se False, poda apenas W_out (que tem omega explícito).
    """

    def __init__(
        self,
        threshold: float = 0.01,
        target_sparsity: float = 0.80,
        prune_rec: bool = True,
    ):
        self.threshold = threshold
        self.target_sparsity = target_sparsity
        self.prune_rec = prune_rec

    def prune(self, brain, verbose: bool = True) -> dict:
        """Executa poda no brain.

        Parameters
        ----------
        brain : VisaoBrain
            Instância do cérebro VISÃO.
        verbose : bool
            Se True, imprime estatísticas.

        Returns
        -------
        dict com estatísticas da poda:
          - pruned_out: número de sinapses W_out podadas
          - pruned_rec: número de sinapses W_rec podadas
          - pruned_total: total de sinapses podadas
          - sparsity_before: sparsity antes da poda
          - sparsity_after: sparsity depois da poda
          - threshold: limiar usado
          - target_sparsity: sparsity alvo
        """
        sparsity_before = compute_sparsity(brain)

        pruned_out = self._prune_wout(brain)
        pruned_rec = self._prune_wrec(brain) if self.prune_rec else 0

        sparsity_after = compute_sparsity(brain)

        stats = {
            "pruned_out": pruned_out,
            "pruned_rec": pruned_rec,
            "pruned_total": pruned_out + pruned_rec,
            "sparsity_before": sparsity_before["total_sparsity"],
            "sparsity_after": sparsity_after["total_sparsity"],
            "threshold": self.threshold,
            "target_sparsity": self.target_sparsity,
        }

        if verbose:
            print(f"✂️  Poda concluída:")
            print(f"   W_out podadas: {pruned_out}")
            print(f"   W_rec podadas: {pruned_rec}")
            print(f"   Sparsity: {sparsity_before['total_sparsity']:.2%} → "
                  f"{sparsity_after['total_sparsity']:.2%}")
            print(f"   Target: {self.target_sparsity:.2%}")

        return stats

    def _prune_wout(self, brain) -> int:
        """Poda sinapses do W_out com omega abaixo do limiar.

        Usa omega como importância. Sinapses com omega < threshold são
        candidatas. Podar apenas até atingir target_sparsity.
        """
        learner = brain.learner
        W = learner.W_out
        omega = learner.omega

        # Candidatos: omega abaixo do limiar E peso não-zero
        candidates = (omega < self.threshold) & (W != 0)

        if not np.any(candidates):
            return 0

        # Ordenar candidatos por importância (menor primeiro)
        candidate_indices = np.argwhere(candidates)
        candidate_importance = omega[candidates]
        sorted_order = np.argsort(candidate_importance)
        candidate_indices = candidate_indices[sorted_order]

        # Calcular quantas podas podemos fazer sem ultrapassar target
        current_sparsity = compute_sparsity(brain)
        total_params = current_sparsity["total_params"]
        current_zeros = current_sparsity["total_zeros"]
        max_zeros = int(self.target_sparsity * total_params)
        budget = max_zeros - current_zeros  # quantas sinapses podemos zerar

        if budget <= 0:
            return 0

        # Podar até o budget
        n_prune = min(len(candidate_indices), budget)
        for i in range(n_prune):
            row, col = candidate_indices[i]
            W[row, col] = 0.0
            # omega também é zerado para manter consistência
            omega[row, col] = 0.0

        return n_prune

    def _prune_wrec(self, brain) -> int:
        """Poda sinapses do W_rec com |W_rec| abaixo do limiar.

        Usa |W_rec| como proxy de importância (não há omega por sinapse
        no reservatório). Respeita a máscara estrutural existente.
        """
        cell = brain.cell
        W = cell.W_rec
        mask = cell.mask

        # Candidatos: |W| abaixo do limiar, peso não-zero, e máscara ativa
        # (não podar onde já está mascarado)
        candidates = (np.abs(W) < self.threshold) & (W != 0) & (mask > 0)

        if not np.any(candidates):
            return 0

        # Ordenar candidatos por importância (menor |W| primeiro)
        candidate_indices = np.argwhere(candidates)
        candidate_importance = np.abs(W[candidates])
        sorted_order = np.argsort(candidate_importance)
        candidate_indices = candidate_indices[sorted_order]

        # Calcular budget
        current_sparsity = compute_sparsity(brain)
        total_params = current_sparsity["total_params"]
        current_zeros = current_sparsity["total_zeros"]
        max_zeros = int(self.target_sparsity * total_params)
        budget = max_zeros - current_zeros

        if budget <= 0:
            return 0

        # Podar até o budget
        n_prune = min(len(candidate_indices), budget)
        for i in range(n_prune):
            row, col = candidate_indices[i]
            W[row, col] = 0.0

        return n_prune


# ==============================================================
#  PODADOR COM PERSISTÊNCIA TEMPORAL
# ==============================================================

class PersistentSynapsePruner(SynapsePruner):
    """Poda sinapses que ficaram com omega baixo por muito tempo.

    Extende SynapsePruner com um mecanismo de "paciência":
    só poda sinapses que ficaram abaixo do limiar por `patience`
    verificações consecutivas.

    Isso evita podar sinapses temporariamente inativas mas que podem
    ser úteis no futuro (princípio: omega baixo por muito tempo = inútil).
    """

    def __init__(
        self,
        threshold: float = 0.01,
        target_sparsity: float = 0.80,
        prune_rec: bool = True,
        patience: int = 10,
    ):
        super().__init__(threshold, target_sparsity, prune_rec)
        self.patience = patience
        self._low_counter = None  # contador de passos abaixo do limiar

    def prune(self, brain, verbose: bool = True) -> dict:
        """Executa poda com verificação de persistência.

        A cada chamada, atualiza contadores de sinapses abaixo do limiar.
        Só poda aquelas que excederam a paciência.
        """
        learner = brain.learner
        W = learner.W_out
        omega = learner.omega

        # Inicializar contador na primeira chamada
        if self._low_counter is None:
            self._low_counter = np.zeros_like(omega)

        # Atualizar contadores
        low_mask = (omega < self.threshold) & (W != 0)
        self._low_counter[low_mask] += 1
        self._low_counter[~low_mask] = 0  # reset se subiu acima do limiar

        # Candidatos: abaixo do limiar por >= patience E não-zero
        candidates = (self._low_counter >= self.patience) & (W != 0)

        if not np.any(candidates):
            if verbose:
                print(f"⏳ Nenhuma sinapse excedeu a paciência ({self.patience})")
            return {
                "pruned_out": 0,
                "pruned_rec": 0,
                "pruned_total": 0,
                "sparsity_before": compute_sparsity(brain)["total_sparsity"],
                "sparsity_after": compute_sparsity(brain)["total_sparsity"],
                "threshold": self.threshold,
                "target_sparsity": self.target_sparsity,
            }

        # Ordenar por importância (menor primeiro)
        candidate_indices = np.argwhere(candidates)
        candidate_importance = omega[candidates]
        sorted_order = np.argsort(candidate_importance)
        candidate_indices = candidate_indices[sorted_order]

        # Calcular budget
        current_sparsity = compute_sparsity(brain)
        total_params = current_sparsity["total_params"]
        current_zeros = current_sparsity["total_zeros"]
        max_zeros = int(self.target_sparsity * total_params)
        budget = max_zeros - current_zeros

        if budget <= 0:
            return {
                "pruned_out": 0,
                "pruned_rec": 0,
                "pruned_total": 0,
                "sparsity_before": current_sparsity["total_sparsity"],
                "sparsity_after": current_sparsity["total_sparsity"],
                "threshold": self.threshold,
                "target_sparsity": self.target_sparsity,
            }

        # Podar até o budget
        n_prune = min(len(candidate_indices), budget)
        for i in range(n_prune):
            row, col = candidate_indices[i]
            W[row, col] = 0.0
            omega[row, col] = 0.0
            self._low_counter[row, col] = 0

        pruned_rec = self._prune_wrec(brain) if self.prune_rec else 0

        sparsity_after = compute_sparsity(brain)

        stats = {
            "pruned_out": n_prune,
            "pruned_rec": pruned_rec,
            "pruned_total": n_prune + pruned_rec,
            "sparsity_before": current_sparsity["total_sparsity"],
            "sparsity_after": sparsity_after["total_sparsity"],
            "threshold": self.threshold,
            "target_sparsity": self.target_sparsity,
        }

        if verbose:
            print(f"✂️  Poda persistente concluída (patience={self.patience}):")
            print(f"   W_out podadas: {n_prune}")
            print(f"   W_rec podadas: {pruned_rec}")
            print(f"   Sparsity: {current_sparsity['total_sparsity']:.2%} → "
                  f"{sparsity_after['total_sparsity']:.2%}")

        return stats

    def reset(self) -> None:
        """Reseta contadores de persistência."""
        self._low_counter = None
