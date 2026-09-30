"""Transformations média classiques d'un MMM : adstock géométrique (effet retard) et saturation de Hill."""
import numpy as np


def adstock(x, decay: float, max_lag: int = 8):
    w = decay ** np.arange(max_lag + 1)
    w = w / w.sum()                                          # conserve le volume total d'investissement
    return np.convolve(x, w)[: len(x)]


def hill(x, half: float, slope: float = 1.5):
    x = np.maximum(x, 0)
    return x ** slope / (x ** slope + half ** slope)
