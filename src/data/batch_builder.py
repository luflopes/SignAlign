"""
Construção de batches para treinamento contrastivo.

Garante que cada batch tenha nomes únicos para evitar
problemas com a loss contrastiva.

Suporta split fixo (v2) com treino/validação/teste.
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset_pairs(
    csv_path: str,
    exclude_unknown: bool = True,
    images_base_path: str = "",
    max_samples: int = None
) -> List[Tuple[str, str]]:
    """
    Carrega pares (imagem, nome) de um CSV.
    
    Args:
        csv_path: Caminho para o arquivo CSV.
        exclude_unknown: Se True, exclui entradas com "UNKNOWN".
        images_base_path: Prefixo opcional para caminhos de imagem.
        max_samples: Se definido, limita o número de amostras (modo teste).
        
    Returns:
        Lista de tuplas (caminho_imagem, nome).
    """
    df = pd.read_csv(csv_path)
    
    if exclude_unknown:
        df = df[df["human_name"] != "UNKNOWN"].reset_index(drop=True)
    
    # Modo de teste: limitar amostras
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
        print(f"⚠️ MODO TESTE: Usando apenas {len(df)} amostras")
    
    pairs = []
    for _, row in df.iterrows():
        img_path = row["image_path"]
        if images_base_path:
            img_path = f"{images_base_path}/{img_path}"
        pairs.append((img_path, row["human_name"]))
    
    return pairs


def load_split_file(split_path: str) -> Dict:
    """
    Carrega arquivo de split JSON.
    
    Args:
        split_path: Caminho para o arquivo JSON do split.
        
    Returns:
        Dicionário com dados do split.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pairs_from_split(
    csv_path: str,
    split_path: str,
    split_name: str,
    images_base_path: str = "",
    exclude_unknown: bool = True,
    max_samples: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Carrega pares para um conjunto específico do split fixo.
    
    Args:
        csv_path: Caminho para o CSV do dataset.
        split_path: Caminho para o JSON do split.
        split_name: Nome do conjunto ("train", "val", "test").
        images_base_path: Prefixo para caminhos de imagem.
        exclude_unknown: Se True, exclui UNKNOWN.
        max_samples: Limita número de amostras (modo teste).
        
    Returns:
        Lista de tuplas (caminho_imagem, nome).
    """
    # Carregar split
    split_data = load_split_file(split_path)
    
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
    
    # Modo teste
    if max_samples is not None and max_samples > 0:
        df_filtered = df_filtered.head(max_samples)
        print(f"⚠️ MODO TESTE: Usando apenas {len(df_filtered)} amostras de {split_name}")
    
    # Criar pares
    pairs = []
    for _, row in df_filtered.iterrows():
        img_path = row["image_path"]
        if images_base_path:
            img_path = f"{images_base_path}/{img_path}"
        pairs.append((img_path, row["human_name"]))
    
    return pairs


def create_train_val_test_split_from_file(
    csv_path: str,
    split_path: str,
    images_base_path: str = "",
    exclude_unknown: bool = True,
    max_samples: Optional[int] = None
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Carrega split fixo de treino/validação/teste de arquivo.
    
    Args:
        csv_path: Caminho para o CSV do dataset.
        split_path: Caminho para o JSON do split.
        images_base_path: Prefixo para caminhos de imagem.
        exclude_unknown: Se True, exclui UNKNOWN.
        max_samples: Limita número de amostras (modo teste).
        
    Returns:
        Tupla (train_pairs, val_pairs, test_pairs).
    """
    train_pairs = load_pairs_from_split(
        csv_path, split_path, "train",
        images_base_path, exclude_unknown, max_samples
    )
    val_pairs = load_pairs_from_split(
        csv_path, split_path, "val",
        images_base_path, exclude_unknown, None  # Não limitar val/test
    )
    test_pairs = load_pairs_from_split(
        csv_path, split_path, "test",
        images_base_path, exclude_unknown, None
    )
    
    # Carregar estatísticas do split
    split_data = load_split_file(split_path)
    
    print(f"📊 Split carregado de: {split_path}")
    print(f"   - Treino:     {len(train_pairs):5d} amostras ({len(split_data['train_individuals']):4d} indivíduos)")
    print(f"   - Validação:  {len(val_pairs):5d} amostras ({len(split_data['val_individuals']):4d} indivíduos)")
    print(f"   - Teste:      {len(test_pairs):5d} amostras ({len(split_data['test_individuals']):4d} indivíduos)")
    
    return train_pairs, val_pairs, test_pairs


def create_train_val_split(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Divide pares em treino/validação por nome (não por amostra).
    
    DEPRECATED: Use create_train_val_test_split_from_file() com split fixo.
    
    Garante que assinaturas do mesmo indivíduo não apareçam
    em ambos os conjuntos.
    
    Args:
        pairs: Lista de (caminho_imagem, nome).
        val_ratio: Proporção do conjunto de validação.
        seed: Seed para reprodutibilidade.
        
    Returns:
        Tupla (train_pairs, val_pairs).
    """
    # Agrupar por nome
    by_name = defaultdict(list)
    for img_path, name in pairs:
        by_name[name].append(img_path)
    
    # Obter nomes únicos
    unique_names = list(by_name.keys())
    
    # Split por nomes
    train_names, val_names = train_test_split(
        unique_names,
        test_size=val_ratio,
        random_state=seed
    )
    
    # Criar pares de treino e validação
    train_pairs = [
        (img, name)
        for name in train_names
        for img in by_name[name]
    ]
    
    val_pairs = [
        (img, name)
        for name in val_names
        for img in by_name[name]
    ]
    
    print(f"📊 Split de dados (legado):")
    print(f"   - Treino: {len(train_pairs)} amostras ({len(train_names)} indivíduos)")
    print(f"   - Validação: {len(val_pairs)} amostras ({len(val_names)} indivíduos)")
    
    # Verificação
    common = set(train_names).intersection(set(val_names))
    if common:
        print(f"⚠️ AVISO: {len(common)} nomes em comum!")
    else:
        print("✅ Nenhum indivíduo em comum entre treino e validação.")
    
    return train_pairs, val_pairs


def build_unique_name_batches(
    pairs: List[Tuple[str, str]],
    batch_size: int = 16,
    shuffle: bool = True
) -> List[List[Tuple[str, str]]]:
    """
    Constrói batches onde nenhum nome se repete.
    
    Essencial para contrastive learning, onde cada item do batch
    deve ser uma classe diferente.
    
    Args:
        pairs: Lista de (caminho_imagem, nome).
        batch_size: Tamanho máximo de cada batch.
        shuffle: Se True, embaralha os pares antes.
        
    Returns:
        Lista de batches, cada batch é uma lista de pares.
    """
    # Agrupar por nome
    by_name = defaultdict(list)
    for img, name in pairs:
        by_name[name].append(img)
    
    # Criar todos os pares possíveis
    all_items = [
        (img, name)
        for name, imgs in by_name.items()
        for img in imgs
    ]
    
    if shuffle:
        random.shuffle(all_items)
    
    batches = []
    used_names = set()
    current_batch = []
    
    for img, name in all_items:
        if name in used_names:
            continue
        
        current_batch.append((img, name))
        used_names.add(name)
        
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
            used_names.clear()
    
    # Adicionar batch final se não estiver vazio
    if current_batch:
        batches.append(current_batch)
    
    return batches


def create_fixed_evaluation_data(
    val_pairs: List[Tuple[str, str]],
    max_negative_samples: int = 3,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Cria dados de avaliação fixos para métricas consistentes.
    
    Para cada nome único, seleciona uma imagem positiva fixa
    e um pool fixo de imagens negativas.
    
    Args:
        val_pairs: Pares de validação.
        max_negative_samples: Máximo de amostras negativas por nome.
        seed: Seed para seleção determinística.
        
    Returns:
        Dicionário {nome: {positive_img, negative_img_pool}}.
    """
    random.seed(seed)
    
    # Agrupar por nome
    by_name = defaultdict(list)
    for img_path, name in val_pairs:
        by_name[name].append(img_path)
    
    unique_names = list(by_name.keys())
    fixed_data = {}
    
    for name in unique_names:
        matching_imgs = by_name[name]
        if not matching_imgs:
            continue
        
        # Selecionar imagem positiva fixa
        positive_img = random.choice(matching_imgs)
        
        # Selecionar pool de imagens negativas
        other_names = [n for n in unique_names if n != name]
        all_negative_imgs = [
            img
            for neg_name in other_names
            for img in by_name[neg_name]
        ]
        
        num_to_sample = min(len(all_negative_imgs), max_negative_samples)
        negative_pool = random.sample(all_negative_imgs, num_to_sample) if num_to_sample > 0 else []
        
        fixed_data[name] = {
            "positive_img": positive_img,
            "negative_img_pool": negative_pool
        }
    
    return fixed_data

