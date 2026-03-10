#!/usr/bin/env python3
"""Lance plusieurs runs (seuils et seeds différents) et affiche un résumé des métriques."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")
BACKTEST_DIR = Path("backtest_results")


def run_one(threshold: float, seed: int, output_subdir: str) -> dict:
    """Modifie la config, lance le script, retourne les métriques."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["threshold"] = threshold
    cfg["model"]["random_state"] = seed
    out_dir = BACKTEST_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output"]["output_dir"] = str(out_dir)
    run_config = out_dir / "config.yaml"
    with open(run_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    r = subprocess.run(
        [sys.executable, "terminal_cluster_model.py", "--config", str(run_config)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
        timeout=120,
    )
    if r.returncode != 0:
        return {"error": r.stderr or str(r.returncode)}

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return {"error": "metrics.json not found"}
    with open(metrics_path) as f:
        return json.load(f)


def main():
    BACKTEST_DIR.mkdir(exist_ok=True)
    results = []

    # Backtest 1: seuils différents (seed fixe 42)
    for th in [0.05, 0.10, 0.20, 0.5]:
        name = f"threshold_{th}"
        print(f"Run {name} ...", flush=True)
        m = run_one(threshold=th, seed=42, output_subdir=name)
        m["run"] = name
        results.append(m)

    # Backtest 2: deux seeds supplémentaires avec seuil 0.1
    for seed in [0, 123]:
        name = f"seed_{seed}_th0.1"
        print(f"Run {name} ...", flush=True)
        m = run_one(threshold=0.1, seed=seed, output_subdir=name)
        m["run"] = name
        results.append(m)

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ BACKTEST")
    print("=" * 60)
    for m in results:
        run = m.get("run", "?")
        if "error" in m:
            print(f"  {run}: ERREUR - {m['error'][:80]}")
            continue
        roc = m.get("roc_auc")
        ap = m.get("average_precision")
        p = m.get("precision")
        r = m.get("recall")
        f1 = m.get("f1")
        cm = m.get("confusion_matrix", {})
        tp, fn = cm.get("tp", 0), cm.get("fn", 0)
        fp, tn = cm.get("fp", 0), cm.get("tn", 0)
        print(f"  {run}: roc_auc={roc:.4f} | AP={ap:.4f} | P={p:.4f} R={r:.4f} F1={f1:.4f} | TP={tp} FP={fp} FN={fn} TN={tn}")
    print("=" * 60)


if __name__ == "__main__":
    main()
