"""
test_multi_timescale.py — Teste do Multi-Timescale Memory.

Demonstra que informações em decaimento rápido (short τ=0.1)
coexistem com memórias persistentes (long τ=10.0).
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visao.memory.multi_timescale import (
    ThreeTimescaleState,
    MultiTimescaleMemory,
)


def test_three_timescale_decay():
    """Testa que cada grupo decai a taxas diferentes."""
    print("=" * 60)
    print("TESTE 1: Decaimento Multi-Timescale")
    print("=" * 60)

    ts = ThreeTimescaleState(
        dims=(8, 8, 8),
        taus=(0.1, 1.0, 10.0),
        dt=0.05,
        seed=42,
    )

    # Injetar padrão idêntico em todos os grupos
    pattern = np.ones(8) * 0.8
    ts.inject("short", pattern, strength=1.0)
    ts.inject("medium", pattern, strength=1.0)
    ts.inject("long", pattern, strength=1.0)

    print(f"\n  Estado inicial (após injeção):")
    print(f"    Short  (τ=0.1): energy={ts.short.energy:.4f}")
    print(f"    Medium (τ=1.0): energy={ts.medium.energy:.4f}")
    print(f"    Long   (τ=10.0): energy={ts.long.energy:.4f}")

    # Deixar decair sem input (100 passos × dt=0.05 = 5.0 unidades de tempo)
    n_steps = 100
    energies = {"short": [], "medium": [], "long": []}
    for _ in range(n_steps):
        ts.decay_step()
        energies["short"].append(ts.short.energy)
        energies["medium"].append(ts.medium.energy)
        energies["long"].append(ts.long.energy)

    print(f"\n  Após {n_steps} passos de decaimento ({n_steps * ts.dt:.1f} time units):")
    print(f"    Short  (τ=0.1): energy={ts.short.energy:.6f} (decaiu ~{energies['short'][0]/max(ts.short.energy,1e-10):.0f}x)")
    print(f"    Medium (τ=1.0): energy={ts.medium.energy:.6f}")
    print(f"    Long   (τ=10.0): energy={ts.long.energy:.6f} (reteve ~{ts.long.energy/energies['long'][0]*100:.1f}%)")

    # Verificar hierarquia: long > medium > short
    assert ts.long.energy > ts.medium.energy > ts.short.energy, \
        "Hierarquia de decaimento violada: long deve reter mais que medium, que retém mais que short"

    # Verificar que short decaiu significativamente (5 time units / 0.1 tau = 50 e-folds)
    assert ts.short.energy < 0.01 * energies["short"][0], \
        "Short deveria ter decaído mais de 99% em 50 e-folds"

    # Verificar que long reteve a maioria (5 time units / 10 tau = 0.5 e-folds → ~60%)
    assert ts.long.energy > 0.5 * energies["long"][0], \
        "Long deveria ter retido mais de 50% em 0.5 e-folds"

    print("\n  ✅ Decaimento multi-temporal OK\n")


def test_fast_decay_persistent_memory_coexistence():
    """Testa coexistência de memória rápida decaída com persistente."""
    print("=" * 60)
    print("TESTE 2: Coexistência Rápido vs Persistente")
    print("=" * 60)

    ts = ThreeTimescaleState(
        dims=(16, 16, 16),
        taus=(0.1, 1.0, 10.0),
        dt=0.05,
        seed=123,
    )

    # Injetar memória temporária em short
    temp_pattern = np.sin(np.linspace(0, 2*np.pi, 16))
    ts.inject("short", temp_pattern, strength=1.0)

    # Injetar memória persistente em long
    perm_pattern = np.cos(np.linspace(0, 4*np.pi, 16))
    ts.inject("long", perm_pattern, strength=1.0)

    print(f"\n  Memória temporária em SHORT (τ=0.1)")
    print(f"  Memória persistente em LONG (τ=10.0)")

    # Snapshot inicial
    ret_short_init = ts.retrieval_strength("short", temp_pattern)
    ret_long_init = ts.retrieval_strength("long", perm_pattern)
    print(f"\n  Retenção inicial:")
    print(f"    Short: {ret_short_init:.4f}")
    print(f"    Long:  {ret_long_init:.4f}")

    # Simular passagem do tempo
    time_points = [10, 30, 50, 100, 150]
    print(f"\n  Evolução temporal:")
    print(f"    {'Step':<8} {'Time':<8} {'Short (τ=0.1)':<18} {'Long (τ=10.0)':<18} {'Ratio L/S':<12}")
    print(f"    {'-'*64}")

    for target_step in time_points:
        # Avançar até target_step
        while ts.step_count < target_step:
            ts.decay_step()

        ret_short = ts.retrieval_strength("short", temp_pattern)
        ret_long = ts.retrieval_strength("long", perm_pattern)
        ratio = ret_long / max(ret_short, 1e-10)
        print(f"    {target_step:<8} {target_step * ts.dt:<8.2f} {ret_short:<18.6f} {ret_long:<18.6f} {ratio:<12.1f}")

    # Verificar coexistência: long mantém memória enquanto short esquece
    ret_short_final = ts.retrieval_strength("short", temp_pattern)
    ret_long_final = ts.retrieval_strength("long", perm_pattern)

    print(f"\n  Resultado final (step {ts.step_count}, t={ts.step_count * ts.dt:.1f}):")
    print(f"    Short (temporária): {ret_short_final:.6f} (decaída)")
    print(f"    Long (persistente): {ret_long_final:.6f} (retida)")

    assert ret_long_final > ret_short_final, \
        "Memória persistente deve ter maior retenção que temporária"
    assert ret_long_final > 0.3, \
        "Memória persistente deve manter >30% de retenção"
    assert ret_short_final < 0.1, \
        "Memória temporária deve ter decaído significativamente"

    print("\n  ✅ Coexistência rápida/persistente OK\n")


def test_multi_timescale_memory_read_write():
    """Testa a interface de read/write com consolidação."""
    print("=" * 60)
    print("TESTE 3: MultiTimescaleMemory (Read/Write/Consolidate)")
    print("=" * 60)

    mtm = MultiTimescaleMemory(
        state_dim=16,
        input_dim=8,
        taus=(0.1, 1.0, 10.0),
        dt=0.05,
        seed=42,
    )

    rng = np.random.default_rng(0)

    # Escrever memórias em diferentes tempos
    mem_short = rng.normal(0, 1, 16)
    mem_medium = rng.normal(0, 1, 16)
    mem_long = rng.normal(0, 1, 16)

    mtm.write("temporary_info", mem_short, timescale="short", strength=1.0)
    mtm.write("daily_knowledge", mem_medium, timescale="medium", strength=1.0)
    mtm.write("core_belief", mem_long, timescale="long", strength=1.0)

    print(f"\n  3 memórias escritas:")
    print(f"    'temporary_info'  → short  (τ=0.1)")
    print(f"    'daily_knowledge' → medium (τ=1.0)")
    print(f"    'core_belief'     → long   (τ=10.0)")

    # Verificar info
    info = mtm.get_memory_info()
    print(f"\n  Info das memórias:")
    for name, data in info.items():
        print(f"    {name}: τ={data['tau']}, retention={data['retention']:.4f}")

    # Simular passagem do tempo (50 passos × dt=0.05 = 2.5 time units)
    print(f"\n  Simulando 50 passos de decaimento (t=2.5)...")
    mtm.rest_step(n_steps=50)

    info_after = mtm.get_memory_info()
    print(f"\n  Info após decaimento:")
    for name, data in info_after.items():
        print(f"    {name}: τ={data['tau']}, retention={data['retention']:.6f}")

    # Verificar hierarquia de retenção
    ret_short = info_after["temporary_info"]["retention"]
    ret_medium = info_after["daily_knowledge"]["retention"]
    ret_long = info_after["core_belief"]["retention"]

    assert ret_long > ret_short, \
        "Memória em long deve ter maior retenção que em short"
    print(f"\n  Long retenção ({ret_long:.4f}) > Short retenção ({ret_short:.4f}) ✓")

    # Testar consolidação
    print(f"\n  Consolidando 'temporary_info' de short → medium...")
    mtm.consolidate("temporary_info")
    info_cons = mtm.get_memory_info()
    print(f"    Novo timescale: {info_cons['temporary_info']['timescale']}")
    assert info_cons["temporary_info"]["timescale"] == "medium", \
        "Deveria ter sido consolidado para medium"

    print(f"\n  Consolidando 'temporary_info' de medium → long...")
    mtm.consolidate("temporary_info")
    info_cons2 = mtm.get_memory_info()
    print(f"    Novo timescale: {info_cons2['temporary_info']['timescale']}")
    assert info_cons2["temporary_info"]["timescale"] == "long", \
        "Deveria ter sido consolidado para long"

    print("\n  ✅ Read/Write/Consolidate OK\n")


def test_dynamic_input_tracking():
    """Testa que short grupo rastreia input dinâmico enquanto long mantém média."""
    print("=" * 60)
    print("TESTE 4: Rastreamento Dinâmico vs Média Persistente")
    print("=" * 60)

    ts = ThreeTimescaleState(
        dims=(8, 8, 8),
        taus=(0.1, 1.0, 10.0),
        dt=0.05,
        seed=42,
    )

    rng = np.random.default_rng(99)

    # Simular input oscilante
    n_steps = 300
    inputs = []
    short_responses = []
    long_responses = []

    for i in range(n_steps):
        # Input com componente rápida + lenta
        fast = np.sin(2 * np.pi * i / 10) * np.ones(8)  # oscila rápido
        slow = np.sin(2 * np.pi * i / 200) * np.ones(8)  # oscila lento
        u = (fast + slow) * 0.5 + rng.normal(0, 0.1, 8)

        ts.step(u)
        inputs.append(u.copy())
        short_responses.append(ts.short.state.copy())
        long_responses.append(ts.long.state.copy())

    # Calcular correlação com input rápido vs lento
    short_arr = np.array(short_responses)
    long_arr = np.array(long_responses)

    # Variância do short deve ser maior (rastreia mudanças rápidas)
    short_var = np.mean(np.var(short_arr, axis=0))
    long_var = np.mean(np.var(long_arr, axis=0))

    print(f"\n  Variância média das respostas:")
    print(f"    Short (τ=0.1): {short_var:.6f} (rastreia rápido)")
    print(f"    Long  (τ=10.0): {long_var:.6f} (suaviza)")

    assert short_var > long_var, \
        "Short deve ter maior variância (rastreia input dinâmico)"

    print("\n  ✅ Rastreamento dinâmico OK\n")


if __name__ == "__main__":
    print("\n🧠 TESTE DO MULTI-TIMESCALE MEMORY\n")

    test_three_timescale_decay()
    test_fast_decay_persistent_memory_coexistence()
    test_multi_timescale_memory_read_write()
    test_dynamic_input_tracking()

    print("=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)
