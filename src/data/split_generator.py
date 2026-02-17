"""
Gerador de split balanceado para dataset de assinaturas.

Cria split fixo (treino/validação/teste) por indivíduo,
balanceando a distribuição de amostras entre os conjuntos.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def count_samples_per_individual(pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Conta o número de amostras por indivíduo.
    
    Args:
        pairs: Lista de (caminho_imagem, nome).
        
    Returns:
        Dicionário {nome: contagem}.
    """
    counts = defaultdict(int)
    for _, name in pairs:
        counts[name] += 1
    return dict(counts)


def stratified_split_by_frequency(
    individual_counts: Dict[str, int],
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Divide indivíduos em treino/validação/teste de forma estratificada.
    
    Estratégia:
    1. Ordena indivíduos por frequência (número de assinaturas)
    2. Divide em bins por frequência
    3. De cada bin, distribui proporcionalmente para train/val/test
    
    Isso garante que cada conjunto tenha uma distribuição similar
    de indivíduos frequentes e raros.
    
    Args:
        individual_counts: Dicionário {nome: contagem}.
        train_ratio: Proporção para treino.
        val_ratio: Proporção para validação.
        test_ratio: Proporção para teste.
        seed: Seed para reprodutibilidade.
        
    Returns:
        Tupla (train_individuals, val_individuals, test_individuals).
    """
    np.random.seed(seed)
    
    # Ordenar por frequência
    sorted_individuals = sorted(
        individual_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Criar bins por frequência
    # Bin 1: indivíduos com muitas amostras (top 20%)
    # Bin 2: indivíduos com média de amostras (20-60%)
    # Bin 3: indivíduos com poucas amostras (bottom 40%)
    n = len(sorted_individuals)
    
    bin_high = [name for name, _ in sorted_individuals[:int(n * 0.2)]]
    bin_medium = [name for name, _ in sorted_individuals[int(n * 0.2):int(n * 0.6)]]
    bin_low = [name for name, _ in sorted_individuals[int(n * 0.6):]]
    
    train_individuals = []
    val_individuals = []
    test_individuals = []
    
    # Para cada bin, distribuir proporcionalmente
    for bin_list in [bin_high, bin_medium, bin_low]:
        np.random.shuffle(bin_list)
        
        n_bin = len(bin_list)
        n_train = int(n_bin * train_ratio)
        n_val = int(n_bin * val_ratio)
        # n_test = resto
        
        train_individuals.extend(bin_list[:n_train])
        val_individuals.extend(bin_list[n_train:n_train + n_val])
        test_individuals.extend(bin_list[n_train + n_val:])
    
    return train_individuals, val_individuals, test_individuals


def generate_balanced_split(
    csv_path: str,
    output_path: str,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 5932,
    exclude_unknown: bool = True
) -> Dict:
    """
    Gera split balanceado e salva em arquivo JSON.
    
    Args:
        csv_path: Caminho para o CSV do dataset.
        output_path: Caminho para salvar o JSON do split.
        train_ratio: Proporção para treino.
        val_ratio: Proporção para validação.
        test_ratio: Proporção para teste.
        seed: Seed para reprodutibilidade.
        exclude_unknown: Se True, exclui entradas UNKNOWN.
        
    Returns:
        Dicionário com informações do split.
    """
    # Validar proporções
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Proporções devem somar 1.0, mas somam {total_ratio}")
    
    # Carregar dados
    df = pd.read_csv(csv_path)
    
    if exclude_unknown:
        df = df[df["human_name"] != "UNKNOWN"].reset_index(drop=True)
    
    # Criar pares e contar
    pairs = [(row["image_path"], row["human_name"]) for _, row in df.iterrows()]
    individual_counts = count_samples_per_individual(pairs)
    
    print(f"📊 Dataset carregado:")
    print(f"   - Total de amostras: {len(pairs)}")
    print(f"   - Total de indivíduos: {len(individual_counts)}")
    
    # Gerar split estratificado
    train_individuals, val_individuals, test_individuals = stratified_split_by_frequency(
        individual_counts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Verificar que não há vazamento
    train_set = set(train_individuals)
    val_set = set(val_individuals)
    test_set = set(test_individuals)
    
    assert len(train_set & val_set) == 0, "Vazamento treino-validação!"
    assert len(train_set & test_set) == 0, "Vazamento treino-teste!"
    assert len(val_set & test_set) == 0, "Vazamento validação-teste!"
    
    # Calcular estatísticas
    def count_samples(individuals: List[str]) -> int:
        return sum(individual_counts[ind] for ind in individuals)
    
    train_samples = count_samples(train_individuals)
    val_samples = count_samples(val_individuals)
    test_samples = count_samples(test_individuals)
    total_samples = train_samples + val_samples + test_samples
    
    # Criar dicionário do split
    split_data = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "train_individuals": sorted(train_individuals),
        "val_individuals": sorted(val_individuals),
        "test_individuals": sorted(test_individuals),
        "statistics": {
            "total": {
                "individuals": len(individual_counts),
                "samples": len(pairs)
            },
            "train": {
                "individuals": len(train_individuals),
                "samples": train_samples,
                "sample_ratio": train_samples / total_samples
            },
            "val": {
                "individuals": len(val_individuals),
                "samples": val_samples,
                "sample_ratio": val_samples / total_samples
            },
            "test": {
                "individuals": len(test_individuals),
                "samples": test_samples,
                "sample_ratio": test_samples / total_samples
            }
        },
        "frequency_distribution": {
            "train": _get_frequency_stats(train_individuals, individual_counts),
            "val": _get_frequency_stats(val_individuals, individual_counts),
            "test": _get_frequency_stats(test_individuals, individual_counts)
        }
    }
    
    # Salvar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Split gerado e salvo em: {output_path}")
    print(f"\n📊 Estatísticas do split:")
    print(f"   Treino:     {len(train_individuals):4d} indivíduos, {train_samples:5d} amostras ({train_samples/total_samples*100:.1f}%)")
    print(f"   Validação:  {len(val_individuals):4d} indivíduos, {val_samples:5d} amostras ({val_samples/total_samples*100:.1f}%)")
    print(f"   Teste:      {len(test_individuals):4d} indivíduos, {test_samples:5d} amostras ({test_samples/total_samples*100:.1f}%)")
    print(f"\n✅ Nenhum vazamento de indivíduos entre conjuntos!")
    
    return split_data


def _get_frequency_stats(individuals: List[str], counts: Dict[str, int]) -> Dict:
    """Calcula estatísticas de frequência para um conjunto de indivíduos."""
    freqs = [counts[ind] for ind in individuals]
    if not freqs:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    
    return {
        "min": int(np.min(freqs)),
        "max": int(np.max(freqs)),
        "mean": float(np.mean(freqs)),
        "median": float(np.median(freqs))
    }


def load_split(split_path: str) -> Dict:
    """
    Carrega um split de arquivo JSON.
    
    Args:
        split_path: Caminho para o arquivo JSON do split.
        
    Returns:
        Dicionário com dados do split.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_pairs(
    csv_path: str,
    split_path: str,
    split_name: str,
    images_base_path: str = "",
    exclude_unknown: bool = True
) -> List[Tuple[str, str]]:
    """
    Obtém pares (imagem, nome) para um conjunto específico do split.
    
    Args:
        csv_path: Caminho para o CSV do dataset.
        split_path: Caminho para o JSON do split.
        split_name: Nome do conjunto ("train", "val", "test").
        images_base_path: Prefixo para caminhos de imagem.
        exclude_unknown: Se True, exclui UNKNOWN.
        
    Returns:
        Lista de tuplas (caminho_imagem, nome).
    """
    # Carregar split
    split_data = load_split(split_path)
    
    # Obter indivíduos do conjunto
    key = f"{split_name}_individuals"
    if key not in split_data:
        raise ValueError(f"Conjunto '{split_name}' não encontrado no split")
    
    individuals_set = set(split_data[key])
    
    # Carregar CSV
    df = pd.read_csv(csv_path)
    
    if exclude_unknown:
        df = df[df["human_name"] != "UNKNOWN"].reset_index(drop=True)
    
    # Filtrar por indivíduos
    df_filtered = df[df["human_name"].isin(individuals_set)]
    
    # Criar pares
    pairs = []
    for _, row in df_filtered.iterrows():
        img_path = row["image_path"]
        if images_base_path:
            img_path = f"{images_base_path}/{img_path}"
        pairs.append((img_path, row["human_name"]))
    
    return pairs


def validate_split(csv_path: str, split_path: str) -> bool:
    """
    Valida que um split é consistente com o dataset.
    
    Args:
        csv_path: Caminho para o CSV do dataset.
        split_path: Caminho para o JSON do split.
        
    Returns:
        True se válido.
    """
    split_data = load_split(split_path)
    df = pd.read_csv(csv_path)
    df = df[df["human_name"] != "UNKNOWN"].reset_index(drop=True)
    
    dataset_individuals = set(df["human_name"].unique())
    
    train_set = set(split_data["train_individuals"])
    val_set = set(split_data["val_individuals"])
    test_set = set(split_data["test_individuals"])
    
    split_individuals = train_set | val_set | test_set
    
    # Verificar cobertura
    missing = dataset_individuals - split_individuals
    extra = split_individuals - dataset_individuals
    
    if missing:
        print(f"⚠️ {len(missing)} indivíduos no dataset não estão no split")
    if extra:
        print(f"⚠️ {len(extra)} indivíduos no split não estão no dataset")
    
    # Verificar vazamento
    overlaps = []
    if train_set & val_set:
        overlaps.append(f"treino-validação: {len(train_set & val_set)}")
    if train_set & test_set:
        overlaps.append(f"treino-teste: {len(train_set & test_set)}")
    if val_set & test_set:
        overlaps.append(f"validação-teste: {len(val_set & test_set)}")
    
    if overlaps:
        print(f"❌ Vazamentos detectados: {', '.join(overlaps)}")
        return False
    
    print("✅ Split válido!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerar split balanceado")
    parser.add_argument(
        "--csv", 
        default="datasets/dataset-sign-align/dataset.csv",
        help="Caminho para o CSV do dataset"
    )
    parser.add_argument(
        "--output",
        default="datasets/dataset-sign-align/splits/split_v2.json",
        help="Caminho para salvar o split"
    )
    parser.add_argument("--seed", type=int, default=5932, help="Seed")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    
    generate_balanced_split(
        csv_path=args.csv,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

