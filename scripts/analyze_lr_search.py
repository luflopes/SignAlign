#!/usr/bin/env python3
"""
Analisa os resultados da busca de Learning Rate e seleciona a melhor LR por modelo.

Lê os resultados em experiments/lr_search/<exp>/metrics/test_metrics.json,
compara a métrica principal (accuracy_3_neg) e escreve um mapeamento
{model_key: best_lr} em experiments/lr_search/best_lr.json.

Esse arquivo é consumido por generate_experiment_grid.py para gerar o grid
de significância usando a melhor LR de cada modelo.

Uso:
    python scripts/analyze_lr_search.py
    python scripts/analyze_lr_search.py --experiments-dir experiments/lr_search
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Métrica principal usada para selecionar a melhor LR
PRIMARY_METRIC = "accuracy_3_neg"
FALLBACK_METRICS = ["accuracy_2_neg", "accuracy_1_neg", "accuracy"]

# Mapeia tag de LR (sufixo do nome do experimento) para valor numérico
LR_TAG_TO_VALUE = {
    "lr1e4": 1e-4,
    "lr1e5": 1e-5,
    "lr1e6": 1e-6,
}


def parse_experiment_name(name: str):
    """
    Separa o nome do experimento em (model_key, lr_value).

    Ex.: "clip_vit_b32_lr1e5" -> ("clip_vit_b32", 1e-5).
    Retorna (None, None) se não for um experimento de busca de LR.
    """
    match = re.search(r"^(.*)_(lr1e[0-9])$", name)
    if not match:
        return None, None
    model_key = match.group(1)
    lr_tag = match.group(2)
    return model_key, LR_TAG_TO_VALUE.get(lr_tag)


def load_metric(metrics_path: Path) -> Optional[float]:
    """Carrega a métrica principal (com fallbacks) de um test_metrics.json."""
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    for key in [PRIMARY_METRIC] + FALLBACK_METRICS:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    return None


def analyze(experiments_dir: Path) -> Dict:
    """Analisa os experimentos de busca de LR e retorna o resumo."""
    results = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        model_key, lr_value = parse_experiment_name(exp_dir.name)
        if model_key is None or lr_value is None:
            continue

        metric = load_metric(exp_dir / "metrics" / "test_metrics.json")
        if metric is None:
            print(f"⚠️ Sem métricas para {exp_dir.name}")
            continue

        results.append({
            "experiment": exp_dir.name,
            "model_key": model_key,
            "learning_rate": lr_value,
            "metric": metric,
        })

    # Selecionar a melhor LR por modelo
    best_lr: Dict[str, float] = {}
    best_detail: Dict[str, Dict] = {}
    for r in results:
        mk = r["model_key"]
        if mk not in best_detail or r["metric"] > best_detail[mk]["metric"]:
            best_detail[mk] = r
            best_lr[mk] = r["learning_rate"]

    return {
        "primary_metric": PRIMARY_METRIC,
        "all_results": sorted(results, key=lambda x: (x["model_key"], x["learning_rate"])),
        "best_lr": best_lr,
        "best_detail": best_detail,
    }


def main():
    parser = argparse.ArgumentParser(description="Analisa busca de LR")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments/lr_search",
        help="Diretório com os resultados da busca de LR",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho para best_lr.json (default: <experiments-dir>/best_lr.json)",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"❌ Diretório não encontrado: {experiments_dir}")
        sys.exit(1)

    summary = analyze(experiments_dir)

    print("\n" + "=" * 60)
    print(f"📊 Busca de LR (métrica: {summary['primary_metric']})")
    print("=" * 60)

    last_model = None
    for r in summary["all_results"]:
        if r["model_key"] != last_model:
            print(f"\n{r['model_key']}:")
            last_model = r["model_key"]
        marker = "  🏆" if summary["best_lr"].get(r["model_key"]) == r["learning_rate"] else "    "
        print(f"{marker} lr={r['learning_rate']:.0e}  {summary['primary_metric']}={r['metric']:.4f}")

    print("\n" + "=" * 60)
    print("🏆 Melhor LR por modelo:")
    for mk, lr in summary["best_lr"].items():
        print(f"   {mk}: {lr:.0e}")
    print("=" * 60)

    output_path = Path(args.output) if args.output else experiments_dir / "best_lr.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "primary_metric": summary["primary_metric"],
            "best_lr": summary["best_lr"],
            "best_detail": summary["best_detail"],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Melhor LR salva em: {output_path}")


if __name__ == "__main__":
    main()
