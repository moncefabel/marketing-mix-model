"""MMM : recherche aléatoire des paramètres adstock / saturation, régression ridge à coefficients positifs,
sélection sur une période de validation temporelle, puis évaluation sur 26 semaines jamais vues."""
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .simulate import CHANNELS
from .transforms import adstock, hill


def controls(dates: pd.Series, t0: int = 0) -> np.ndarray:
    w = np.arange(t0, t0 + len(dates))
    q4 = np.isin(dates.dt.month, [11, 12]).astype(float)
    return np.column_stack([w / 52, np.sin(2 * np.pi * w / 52), np.cos(2 * np.pi * w / 52),
                            np.sin(4 * np.pi * w / 52), np.cos(4 * np.pi * w / 52), q4, q4 * w / 52])


def media_matrix(df, params, means):
    return np.column_stack([hill(adstock(df[c].values, params[c][0]), params[c][1] * means[c])
                            for c in CHANNELS])


def design(df, params, means):
    return np.column_stack([media_matrix(df, params, means), controls(df["date"])])


class ConstrainedLinear:
    """Régression linéaire avec effets média contraints positifs (un investissement ne détruit pas de ventes),
    variables de contrôle (tendance, saisonnalité) libres."""
    def __init__(self, n_media: int):
        self.n_media = n_media

    def fit(self, X, y):
        Xb = np.column_stack([X, np.ones(len(X))])
        lb = np.r_[np.zeros(self.n_media), np.full(X.shape[1] - self.n_media + 1, -np.inf)]
        sol = lsq_linear(Xb, y, bounds=(lb, np.full(Xb.shape[1], np.inf)))
        self.coef_, self.intercept_ = sol.x[:-1], sol.x[-1]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def fit(df_train, params, means):
    X, y = design(df_train, params, means), df_train["ventes"].values
    return ConstrainedLinear(len(CHANNELS)).fit(X, y)


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y))


def search(df, n_train: int, n_val: int, n_iter: int = 3000, seed: int = 0, calibration=None, weight=1.0):
    """calibration : {canal: ROAS mesuré par un test d'incrémentalité}. Les jeux de paramètres dont le ROAS
    implicite s'écarte du résultat expérimental sont pénalisés (principe de calibration de Robyn / Meridian)."""
    rng = np.random.default_rng(seed)
    means = {c: df[c].iloc[:n_train].mean() for c in CHANNELS}
    tr, va = df.iloc[:n_train - n_val], df.iloc[:n_train]
    best = (np.inf, None, None)
    for _ in range(n_iter):
        params = {c: (rng.uniform(0, 0.85), rng.uniform(0.3, 3.0)) for c in CHANNELS}
        m = fit(tr, params, means)
        pred = m.predict(design(va, params, means))[-n_val:]
        err = mape(va["ventes"].values[-n_val:], pred)
        penalty = 0.0
        if calibration:
            contrib = contributions(m, tr, params, means)
            penalty = np.mean([abs(contrib[c] / tr[c].sum() - r) / r for c, r in calibration.items()])
        score = err + weight * penalty * 0.1
        if score < best[0]:
            best = (score, params, err)
    return best[1], means, best[2]


def contributions(model, df, params, means):
    M = media_matrix(df, params, means)
    return {c: float(model.coef_[i] * M[:, i].sum()) for i, c in enumerate(CHANNELS)}


def optimise_budget(model, df_period, params, means, bounds=(0.5, 1.5)):
    """Réallocation à budget constant : facteur multiplicatif par canal, borné à +/- 50 %."""
    from scipy.optimize import minimize
    spend_tot = np.array([df_period[c].sum() for c in CHANNELS])

    def media_sales(s):
        d = df_period.copy()
        for i, c in enumerate(CHANNELS):
            d[c] = d[c] * s[i]
        M = media_matrix(d, params, means)
        return float((M * model.coef_[: len(CHANNELS)]).sum())

    cons = {"type": "eq", "fun": lambda s: (s * spend_tot).sum() - spend_tot.sum()}
    res = minimize(lambda s: -media_sales(s), np.ones(len(CHANNELS)), bounds=[bounds] * len(CHANNELS),
                   constraints=cons, method="SLSQP")
    return res.x, media_sales(np.ones(len(CHANNELS))), media_sales(res.x)
