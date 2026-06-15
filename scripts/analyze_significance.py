#!/usr/bin/env python3
"""
Análise de significância estatística do grid multi-seed.

Lê os resultados de experiments/significance/seed_<s>/<config>/metrics/test_metrics.json
e produz:
  1. Agregação por configuração (média, desvio, IC 95%) -> summary.csv
  2. Testes pareados (por seed) para efeitos:
       - augmentação (config vs config_noaug)
       - loss combinada vs finetuning simples (combined_twXX vs infonce/sigmoid)
  3. Comparação entre modelos: Friedman + post-hoc Nemenyi
  -> significance_tests.json
  4. Figura com barras + IC 95% -> significance_accuracy.png

Uso:
    python scripts/analyze_significance.py
    python scripts/analyze_significance.py --base-dir experiments/significance \
        --metric accuracy_3_neg
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Dependências estatísticas (opcionais, mas recomendadas)
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

# Métrica principal e métricas adicionais a agregar
PRIMARY_METRIC = "accuracy_3_neg"
FOCUS_METRICS = [
    "accuracy_3_neg",
    "accuracy_2_neg",
    "accuracy_1_neg",
    "mrr_3_neg",
    "similarity_gap_3_neg",
]

# Modelos conhecidos (prefixos dos nomes de config)
MODEL_PREFIXES = ["clip_vit_b32", "clip_vit_b16", "tinyclip", "siglip"]
# Loss base (finetuning simples) por modelo
BASE_FINETUNE = {
    "tinyclip": "tinyclip_infonce",
    "clip_vit_b32": "clip_vit_b32_infonce",
    "siglip": "siglip_sigmoid",
}


# ----------------------------------------------------------------------------
# Carregamento dos resultados
# ----------------------------------------------------------------------------

def load_results(base_dir: Path) -> pd.DataFrame:
    """
    Varre experiments/significance/seed_<s>/<config>/metrics/test_metrics.json
    e retorna um DataFrame longo: colunas [config, seed, <métricas...>].
    """
    rows = []
    seed_dirs = sorted(base_dir.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(
            f"Nenhum diretório seed_* encontrado em {base_dir}. "
            "Rode scripts/run_significance_grid.sh primeiro."
        )

    for seed_dir in seed_dirs:
        seed_match = re.search(r"seed_(\-?\d+)$", seed_dir.name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        for exp_dir in sorted(seed_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            metrics_path = exp_dir / "metrics" / "test_metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            row = {"config": exp_dir.name, "seed": seed}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"Nenhum test_metrics.json encontrado em {base_dir}/seed_*/*/metrics/."
        )

    return pd.DataFrame(rows)


def parse_model(config: str) -> Optional[str]:
    """Extrai o modelo a partir do nome da config."""
    for prefix in MODEL_PREFIXES:
        if config.startswith(prefix):
            return prefix
    return None


# ----------------------------------------------------------------------------
# Agregação
# ----------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Agrega por configuração: n, média, desvio e IC 95% para cada métrica."""
    available = [m for m in metrics if m in df.columns]
    records = []

    for config, group in df.groupby("config"):
        rec = {"config": config, "model": parse_model(config), "n_seeds": len(group)}
        for m in available:
            vals = group[m].dropna().values
            n = len(vals)
            mean = float(np.mean(vals)) if n > 0 else float("nan")
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            # IC 95% via t-Student
            if n > 1 and HAS_SCIPY:
                tcrit = scipy_stats.t.ppf(0.975, df=n - 1)
                ci = tcrit * std / math.sqrt(n)
            elif n > 1:
                ci = 1.96 * std / math.sqrt(n)
            else:
                ci = 0.0
            rec[f"{m}_mean"] = mean
            rec[f"{m}_std"] = std
            rec[f"{m}_ci95"] = float(ci)
        records.append(rec)

    summary = pd.DataFrame(records)
    sort_col = f"{PRIMARY_METRIC}_mean"
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return summary


# ----------------------------------------------------------------------------
# Testes pareados
# ----------------------------------------------------------------------------

def paired_vectors(df: pd.DataFrame, config_a: str, config_b: str, metric: str):
    """Retorna vetores pareados por seed para duas configs (apenas seeds comuns)."""
    a = df[df["config"] == config_a].set_index("seed")[metric]
    b = df[df["config"] == config_b].set_index("seed")[metric]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return None, None, common
    return a.loc[common].values, b.loc[common].values, common


def paired_test(df: pd.DataFrame, config_a: str, config_b: str, metric: str) -> Optional[Dict]:
    """Executa Wilcoxon signed-rank e t-test pareado entre duas configs."""
    a, b, common = paired_vectors(df, config_a, config_b, metric)
    if a is None or len(common) < 2:
        return None

    diff = a - b
    result = {
        "config_a": config_a,
        "config_b": config_b,
        "metric": metric,
        "n_pairs": len(common),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
    }

    if HAS_SCIPY:
        # t-test pareado
        try:
            t_stat, t_p = scipy_stats.ttest_rel(a, b)
            result["ttest_p"] = float(t_p)
            result["ttest_stat"] = float(t_stat)
        except Exception as e:
            result["ttest_error"] = str(e)

        # Wilcoxon (requer diferenças não todas nulas)
        if np.any(diff != 0):
            try:
                w_stat, w_p = scipy_stats.wilcoxon(a, b)
                result["wilcoxon_p"] = float(w_p)
                result["wilcoxon_stat"] = float(w_stat)
            except Exception as e:
                result["wilcoxon_error"] = str(e)
        else:
            result["wilcoxon_p"] = 1.0

        # Cohen's d para amostras pareadas
        sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
        result["cohens_d"] = float(np.mean(diff) / sd) if sd > 0 else 0.0

    p = result.get("wilcoxon_p", result.get("ttest_p"))
    result["significant_0.05"] = bool(p is not None and p < 0.05)
    return result


def augmentation_effects(df: pd.DataFrame, metric: str) -> List[Dict]:
    """Compara cada config (com aug) vs seu par _noaug."""
    configs = set(df["config"].unique())
    effects = []
    for config in sorted(configs):
        if config.endswith("_noaug"):
            continue
        noaug = f"{config}_noaug"
        if noaug in configs:
            res = paired_test(df, config, noaug, metric)
            if res:
                res["comparison"] = "augmentation_on_vs_off"
                effects.append(res)
    return effects


def combined_loss_effects(df: pd.DataFrame, metric: str) -> List[Dict]:
    """Compara loss combinada (combined_twXX) vs finetuning simples por modelo."""
    configs = set(df["config"].unique())
    effects = []
    for config in sorted(configs):
        if "_combined_tw" not in config or config.endswith("_noaug"):
            continue
        model = parse_model(config)
        base = BASE_FINETUNE.get(model)
        if base and base in configs:
            res = paired_test(df, config, base, metric)
            if res:
                res["comparison"] = "combined_vs_simple_finetune"
                effects.append(res)
    return effects


# ----------------------------------------------------------------------------
# Comparação entre modelos (Friedman + Nemenyi)
# ----------------------------------------------------------------------------

def model_comparison(df: pd.DataFrame, metric: str) -> Dict:
    """
    Friedman test (+ Nemenyi post-hoc) comparando o finetuning simples
    de cada modelo, com as seeds como blocos.
    """
    result = {"metric": metric, "configs": dict(BASE_FINETUNE)}

    # Construir matriz seeds x modelos (apenas seeds comuns a todos)
    per_model = {}
    for model, config in BASE_FINETUNE.items():
        s = df[df["config"] == config].set_index("seed")[metric]
        if len(s) > 0:
            per_model[model] = s

    if len(per_model) < 3:
        result["note"] = "Modelos insuficientes para Friedman (precisa de >= 3)."
        return result

    common_seeds = set.intersection(*[set(s.index) for s in per_model.values()])
    common_seeds = sorted(common_seeds)
    if len(common_seeds) < 2:
        result["note"] = "Seeds comuns insuficientes."
        return result

    models = list(per_model.keys())
    matrix = np.array([[per_model[m].loc[seed] for m in models] for seed in common_seeds])
    result["n_blocks"] = len(common_seeds)
    result["model_means"] = {m: float(np.mean(matrix[:, i])) for i, m in enumerate(models)}

    if HAS_SCIPY:
        try:
            chi, p = scipy_stats.friedmanchisquare(*[matrix[:, i] for i in range(len(models))])
            result["friedman_stat"] = float(chi)
            result["friedman_p"] = float(p)
            result["significant_0.05"] = bool(p < 0.05)
        except Exception as e:
            result["friedman_error"] = str(e)

    if HAS_POSTHOCS:
        try:
            nemenyi = sp.posthoc_nemenyi_friedman(matrix)
            nemenyi.index = models
            nemenyi.columns = models
            result["nemenyi_p"] = nemenyi.to_dict()
        except Exception as e:
            result["nemenyi_error"] = str(e)
    elif not HAS_POSTHOCS:
        result["nemenyi_note"] = "Instale scikit-posthocs para o post-hoc de Nemenyi."

    return result


# ----------------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------------

def plot_summary(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Gera barra horizontal com média + IC 95% por configuração."""
    mean_col = f"{metric}_mean"
    ci_col = f"{metric}_ci95"
    if mean_col not in summary.columns:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib indisponível; pulando figura.")
        return

    data = summary.sort_values(mean_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(data))))
    ax.barh(
        data["config"],
        data[mean_col],
        xerr=data[ci_col] if ci_col in data.columns else None,
        capsize=3,
    )
    ax.set_xlabel(f"{metric} (média ± IC 95%)")
    ax.set_title("Significância estatística - desempenho por configuração")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"🖼️  Figura salva em: {output_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Análise de significância estatística")
    parser.add_argument("--base-dir", type=str, default="experiments/significance")
    parser.add_argument("--metric", type=str, default=PRIMARY_METRIC)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    metric = args.metric

    if not HAS_SCIPY:
        print("⚠️ scipy não instalado: testes estatísticos serão limitados. "
              "Rode: pip install scipy scikit-posthocs")

    df = load_results(base_dir)
    print(f"📥 {len(df)} execuções carregadas "
          f"({df['config'].nunique()} configs, {df['seed'].nunique()} seeds).")

    # 1. Agregação
    summary = aggregate(df, FOCUS_METRICS)
    summary_path = base_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"💾 Resumo salvo em: {summary_path}")

    # 2 + 3. Testes
    tests = {
        "metric": metric,
        "n_seeds": int(df["seed"].nunique()),
        "augmentation_effects": augmentation_effects(df, metric),
        "combined_loss_effects": combined_loss_effects(df, metric),
        "model_comparison": model_comparison(df, metric),
        "backends": {"scipy": HAS_SCIPY, "scikit_posthocs": HAS_POSTHOCS},
    }
    tests_path = base_dir / "significance_tests.json"
    with open(tests_path, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)
    print(f"💾 Testes salvos em: {tests_path}")

    # 4. Figura
    plot_summary(summary, metric, base_dir / "significance_accuracy.png")

    # Resumo no terminal
    print("\n" + "=" * 60)
    print(f"📊 Top configurações ({metric}):")
    print("=" * 60)
    cols = ["config", f"{metric}_mean", f"{metric}_std", f"{metric}_ci95", "n_seeds"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].head(10).to_string(index=False))

    mc = tests["model_comparison"]
    if "friedman_p" in mc:
        print(f"\nFriedman (modelos, {metric}): p={mc['friedman_p']:.4g} "
              f"-> {'significativo' if mc.get('significant_0.05') else 'não significativo'}")

    sig_aug = sum(1 for e in tests["augmentation_effects"] if e.get("significant_0.05"))
    sig_comb = sum(1 for e in tests["combined_loss_effects"] if e.get("significant_0.05"))
    print(f"Efeitos de augmentação significativos: {sig_aug}/{len(tests['augmentation_effects'])}")
    print(f"Efeitos de loss combinada significativos: {sig_comb}/{len(tests['combined_loss_effects'])}")


if __name__ == "__main__":
    main()
