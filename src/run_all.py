"""Point d'entrée : python -m src.run_all
Compare un MMM non calibré et un MMM calibré par un test d'incrémentalité (simulé) sur la TV,
en vérifiant chaque recommandation budgétaire contre la vérité terrain de la simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from . import mmm
from .simulate import CHANNELS, MEAN_SPEND, TRUE, simulate, true_media_sales
from .transforms import adstock, hill

ROOT = Path(__file__).resolve().parents[1]
N_TEST, N_VAL = 26, 26


def true_roas(df, c, end):
    s = df[c].values
    eff = TRUE[c]["beta"] * hill(adstock(s, TRUE[c]["decay"]), TRUE[c]["half"] * MEAN_SPEND[c])
    return float(eff[:end].sum() / s[:end].sum())


def scaled(period, factors):
    d = period.copy()
    for i, c in enumerate(CHANNELS):
        d[c] = d[c] * factors[i]
    return d


def run(df, n_train, calibration=None):
    params, means, val_mape = mmm.search(df, n_train, N_VAL, calibration=calibration)
    model = mmm.fit(df.iloc[:n_train], params, means)
    pred = model.predict(mmm.design(df, params, means))
    contrib = mmm.contributions(model, df.iloc[:n_train], params, means)
    last = df.iloc[-52:].copy()
    s_opt, before, after = mmm.optimise_budget(model, last, params, means)
    return {
        "val_mape": val_mape,
        "test_mape": mmm.mape(df["ventes"].values[n_train:], pred[n_train:]),
        "roas_estimated": {c: contrib[c] / df[c].iloc[:n_train].sum() for c in CHANNELS},
        "budget_factors": dict(zip(CHANNELS, map(float, s_opt))),
        "gain_predicted": after / before - 1,
        "gain_true": true_media_sales(scaled(last, s_opt)) / true_media_sales(last) - 1,
        "_pred": pred,
    }


def main():
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    df, _ = simulate()
    n_train = len(df) - N_TEST

    # Test d'incrémentalité simulé sur la TV : mesure du vrai ROAS avec 5 % de bruit de mesure
    rng = np.random.default_rng(3)
    lift_test = {"tv": true_roas(df, "tv", n_train - N_VAL) * (1 + rng.normal(0, 0.05))}

    raw = run(df, n_train)
    cal = run(df, n_train, calibration=lift_test)

    last = df.iloc[-52:].copy()
    tot = np.array([last[c].sum() for c in CHANNELS])
    base = true_media_sales(last)
    best = minimize(lambda s: -true_media_sales(scaled(last, s)), np.ones(len(CHANNELS)),
                    bounds=[(0.5, 1.5)] * len(CHANNELS), method="SLSQP",
                    constraints={"type": "eq", "fun": lambda s: (s * tot).sum() - tot.sum()})
    max_gain = -best.fun / base - 1

    res = {"lift_test_tv_roas": lift_test["tv"],
           "roas_true": {c: true_roas(df, c, n_train) for c in CHANNELS},
           "max_achievable_gain": max_gain,
           "uncalibrated": {k: v for k, v in raw.items() if not k.startswith("_")},
           "calibrated_tv": {k: v for k, v in cal.items() if not k.startswith("_")}}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(res, indent=2))
    for name in ["uncalibrated", "calibrated_tv"]:
        r = res[name]
        print(f"{name:14s} test MAPE {r['test_mape']:.3f} | gain prédit {r['gain_predicted']:+.1%} "
              f"| gain réel {r['gain_true']:+.1%} | max atteignable {max_gain:+.1%}")

    plt.figure(figsize=(7, 4))
    labels = ["MMM non calibré", "MMM calibré (TV)", "Optimum réel"]
    x = np.arange(3)
    plt.bar(x - 0.2, [raw["gain_predicted"] * 100, cal["gain_predicted"] * 100, np.nan], 0.4, label="Gain prédit par le modèle")
    plt.bar(x + 0.2, [raw["gain_true"] * 100, cal["gain_true"] * 100, max_gain * 100], 0.4, label="Gain réel (vérité terrain)")
    plt.xticks(x, labels); plt.ylabel("Ventes média, variation (%)"); plt.legend()
    plt.title("Réallocation à budget constant : promesse contre réalité"); plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "budget_gain.png", dpi=150); plt.close()

    x = np.arange(len(CHANNELS))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.27, [res["roas_true"][c] for c in CHANNELS], 0.27, label="ROAS réel")
    plt.bar(x, [raw["roas_estimated"][c] for c in CHANNELS], 0.27, label="Non calibré")
    plt.bar(x + 0.27, [cal["roas_estimated"][c] for c in CHANNELS], 0.27, label="Calibré (TV)")
    plt.xticks(x, CHANNELS); plt.ylabel("Ventes par euro investi"); plt.legend()
    plt.title("ROAS par canal : estimation contre vérité"); plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "roas_recovery.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(df["date"], df["ventes"], label="Ventes observées")
    plt.plot(df["date"], cal["_pred"], label="Prédiction MMM calibré")
    plt.axvline(df["date"].iloc[n_train], c="k", ls="--", label="Début du test")
    plt.ylabel("k€ / semaine"); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "fit.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
