#!/usr/bin/env python3
"""
Instancia os três modelos do projeto (TinyCLIP, CLIP-B/32, SigLIP) e conta
a quantidade de parâmetros de cada um (totais e treináveis).
"""

from transformers import AutoModel

MODELS = {
    "TinyCLIP": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
    "CLIP-B/32": "openai/clip-vit-base-patch32",
    "SigLIP": "google/siglip-base-patch16-224",
}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print(f"{'Modelo':<12} {'Total':>15} {'Treináveis':>15}")
    print("-" * 44)
    for label, name in MODELS.items():
        model = AutoModel.from_pretrained(name)
        total, trainable = count_parameters(model)
        print(f"{label:<12} {total:>15,} {trainable:>15,}")
        print(f"{'':<12} {total / 1e6:>13.1f}M {trainable / 1e6:>13.1f}M")


if __name__ == "__main__":
    main()
