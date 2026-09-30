"""Données simulées d'une Maison de luxe fictive, avec vérité terrain connue.
Simuler permet de vérifier qu'un MMM retrouve les vrais effets, ce qui est impossible sur données réelles."""
import numpy as np
import pandas as pd

from .transforms import adstock, hill

CHANNELS = ["tv", "presse", "social", "search", "influence"]
TRUE = {  # decay adstock, demi-saturation (en multiple de la dépense moyenne), effet max (k€ de ventes/sem.)
    # Scénario volontaire : la TV est sur-investie (saturée), le social et l'influence sous-investis.
    "tv":        {"decay": 0.6, "half": 0.5, "beta": 700},
    "presse":    {"decay": 0.5, "half": 1.0, "beta": 300},
    "social":    {"decay": 0.3, "half": 2.0, "beta": 900},
    "search":    {"decay": 0.1, "half": 1.0, "beta": 350},
    "influence": {"decay": 0.4, "half": 1.8, "beta": 600},
}
MEAN_SPEND = {"tv": 120, "presse": 40, "social": 60, "search": 50, "influence": 35}  # k€ / semaine


def spend_pattern(rng, n, mean, flight_prob):
    on = rng.random(n) < flight_prob
    s = np.where(on, rng.gamma(4, mean / (4 * flight_prob), n), 0.0)
    return s


def simulate(n_weeks: int = 156, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    week = np.arange(n_weeks)
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    q4 = np.isin(dates.month, [11, 12]).astype(float)
    season = 1 + 0.25 * np.sin(2 * np.pi * week / 52 - 1.2) + 0.35 * q4    # fêtes de fin d'année
    base = (2500 + 4 * week) * season
    df = pd.DataFrame({"date": dates})
    contrib = {}
    flights = {"tv": 0.45, "presse": 0.6, "social": 0.9, "search": 1.0, "influence": 0.5}
    for c in CHANNELS:
        s = spend_pattern(rng, n_weeks, MEAN_SPEND[c], flights[c]) * (1 + 0.5 * q4)
        df[c] = s
        p = TRUE[c]
        contrib[c] = p["beta"] * hill(adstock(s, p["decay"]), p["half"] * MEAN_SPEND[c])
    media = sum(contrib.values())
    df["ventes"] = base + media + rng.normal(0, 0.04 * base.mean(), n_weeks)
    truth = {c: {"roas": float(contrib[c].sum() / df[c].sum()), "contribution": float(contrib[c].sum())}
             for c in CHANNELS}
    truth["_base_share"] = float(base.sum() / df["ventes"].sum())
    return df, truth


def true_media_sales(spend: pd.DataFrame) -> float:
    return float(sum(TRUE[c]["beta"] * hill(adstock(spend[c].values, TRUE[c]["decay"]),
                                            TRUE[c]["half"] * MEAN_SPEND[c]).sum() for c in CHANNELS))
