"""Testes do benchmark de séries temporais (tarefa 1.5, TDD).

RED primeiro: este arquivo DEVE falhar (módulo inexistente) antes de
visao/bench/timeseries.py ser escrito.
"""
import numpy as np
import pytest


def test_smnist_loader_shapes_and_labels():
    from visao.bench import timeseries
    x_tr, y_tr, x_te, y_te = timeseries.load_smnist_permuted(
        n_train=40, n_test=20, perm_seed=7
    )
    assert x_tr.shape == (40, 784, 1), x_tr.shape
    assert x_te.shape == (20, 784, 1), x_te.shape
    assert set(np.unique(y_tr).tolist()).issubset(set(range(10)))
    assert y_tr.dtype == np.int32


def test_permutation_is_deterministic_and_active():
    from visao.bench import timeseries
    a = timeseries.load_smnist_permuted(n_train=10, n_test=0, perm_seed=7)
    b = timeseries.load_smnist_permuted(n_train=10, n_test=0, perm_seed=7)
    c = timeseries.load_smnist_permuted(n_train=10, n_test=0, perm_seed=99)
    # mesma seed -> idêntico
    assert np.array_equal(a[0], b[0])
    # semente diferente -> ordem dos pixels difere (a permutação é ativa)
    assert not np.array_equal(a[0], c[0])
    # cada amostra ainda contém exatamente os 784 pixels originais
    assert sorted(a[0][:, 0].tolist()) == sorted(c[0][:, 0].tolist())


def test_cfc_trainable_params_below_lstm():
    from visao.bench import timeseries
    cfc_p = timeseries.count_params(timeseries.cfc_params(cfc_hidden=32))
    lstm_p = timeseries.count_params(timeseries.lstm_params(lstm_hidden=128))
    assert cfc_p < lstm_p  # a alegação do projeto: menos parâmetros


def test_gru_trainable_params_below_lstm():
    from visao.bench import timeseries
    gru_p = timeseries.count_params(timeseries.gru_params(gru_hidden=128))
    lstm_p = timeseries.count_params(timeseries.lstm_params(lstm_hidden=128))
    assert gru_p < lstm_p  # GRU tem 3 portas, LSTM 4


def test_gru_forward_runs():
    from visao.bench import timeseries
    import jax.numpy as jnp
    p = timeseries.gru_params(gru_hidden=16)
    out = timeseries.gru_forward(p, jnp.zeros((5, 1)))
    assert out.shape == (10,)


def test_benchmark_runs_and_reports():
    from visao.bench import timeseries
    res = timeseries.benchmark_smnist(
        n_train=128, n_test=64, n_epochs=2,
        cfc_hidden=16, lstm_hidden=32, batch=64, seed=0,
    )
    assert isinstance(res, dict)
    for key in ("cfc", "lstm"):
        assert key in res, res.keys()
        assert 0.0 <= res[key]["accuracy"] <= 1.0
        assert res[key]["params"] > 0
    assert res["cfc"]["params"] < res["lstm"]["params"]
