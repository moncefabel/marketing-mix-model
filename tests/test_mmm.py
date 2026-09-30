import numpy as np

from src.transforms import adstock, hill


def test_adstock_preserves_total_spend_when_series_is_long():
    x = np.r_[np.ones(10), np.zeros(20)]
    assert abs(adstock(x, 0.5).sum() - x.sum()) < 1e-9


def test_adstock_carries_effect_forward():
    x = np.r_[1.0, np.zeros(5)]
    y = adstock(x, 0.6)
    assert y[1] > 0 and y[1] < y[0]


def test_hill_is_half_at_half_saturation():
    assert abs(hill(np.array([2.0]), half=2.0)[0] - 0.5) < 1e-12
