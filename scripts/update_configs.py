#!/usr/bin/env python3
"""
Script para atualizar configurações YAML dos experimentos.

Atualiza:
- epochs: 20 -> 30
- Adiciona split_path para usar split fixo
"""

import yaml
from pathlib import Path
import sys


def update_yaml_config(yaml_path: Path) -> bool:
    """
    Atualiza um arquivo YAML de configuração.
    
    Args:
        yaml_path: Caminho para o arquivo YAML.
        
    Returns:
        True se houve alterações.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    changed = False
    
    # Atualizar epochs
    if "training" in config:
        if config["training"].get("epochs", 20) == 20:
            config["training"]["epochs"] = 30
            changed = True
    
    # Adicionar split_path se não existir
    if "data" in config:
        if "split_path" not in config["data"]:
            config["data"]["split_path"] = "datasets/dataset-sign-align/splits/split_v2.json"
            changed = True
        
        # Atualizar ratios para o novo padrão
        if config["data"].get("val_ratio") == 0.15:
            config["data"]["train_ratio"] = 0.75
            config["data"]["val_ratio"] = 0.10
            config["data"]["test_ratio"] = 0.15
            changed = True
    
    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return changed


def main():
    """Atualiza todas as configurações."""
    configs_dir = Path("configs/grid")
    
    if not configs_dir.exists():
        print(f"❌ Diretório não encontrado: {configs_dir}")
        sys.exit(1)
    
    yaml_files = list(configs_dir.glob("*.yaml"))
    
    print(f"📁 Encontrados {len(yaml_files)} arquivos YAML em {configs_dir}")
    
    updated = 0
    for yaml_path in sorted(yaml_files):
        if update_yaml_config(yaml_path):
            print(f"   ✅ Atualizado: {yaml_path.name}")
            updated += 1
        else:
            print(f"   ⏭️  Sem alterações: {yaml_path.name}")
    
    print(f"\n📊 Resumo: {updated}/{len(yaml_files)} arquivos atualizados")
    
    # Atualizar também configs internos se existirem
    internal_dir = Path("configs/internal")
    if internal_dir.exists():
        internal_files = list(internal_dir.glob("*.yaml"))
        print(f"\n📁 Encontrados {len(internal_files)} arquivos YAML em {internal_dir}")
        
        for yaml_path in sorted(internal_files):
            if update_yaml_config(yaml_path):
                print(f"   ✅ Atualizado: {yaml_path.name}")


if __name__ == "__main__":
    main()

