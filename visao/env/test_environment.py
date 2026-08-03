"""Tarefa 1.1 — ambiente JAX.

Teste de aceite: o JAX deve estar importável e expor ao menos um
dispositivo (CPU, nesta máquina). Espelha o comando de verificação do
BACKLOG: `python3 -c 'import jax; print(jax.devices())'`.
"""


def test_jax_importable_and_has_device():
    import importlib

    jax = importlib.import_module("jax")
    devices = jax.devices()
    assert len(devices) >= 1, "nenhum dispositivo JAX disponível"
