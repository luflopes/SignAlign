"""
Attention Rollout para visualização de atenção em ViT.

Baseado em "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020).
"""

from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from src.data.dataset import paste_center_on_canvas


class AttentionRollout:
    """
    Implementação de Attention Rollout para Vision Transformers.
    
    Combina atenção de múltiplas camadas para visualizar
    quais regiões da imagem são mais relevantes.
    """
    
    def __init__(self, model, processor, device: torch.device):
        """
        Args:
            model: Wrapper do modelo (deve suportar output_attentions).
            processor: Processor do modelo.
            device: Dispositivo de computação.
        """
        self.model = model
        self.processor = processor
        self.device = device
    
    def compute_rollout(
        self,
        attentions: Tuple[torch.Tensor, ...],
        head_fusion: str = "mean"
    ) -> torch.Tensor:
        """
        Computa attention rollout a partir de atenções por camada.
        
        Args:
            attentions: Tupla de tensores de atenção (L, B, H, N, N).
            head_fusion: Método de fusão de heads ("mean", "max", "min").
            
        Returns:
            Attention map (B, N, N).
        """
        # Stack layers: (L, B, H, N, N)
        attn = torch.stack(attentions)
        
        # Fuse heads
        if head_fusion == "mean":
            attn = attn.mean(dim=2)
        elif head_fusion == "max":
            attn = attn.max(dim=2)[0]
        elif head_fusion == "min":
            attn = attn.min(dim=2)[0]
        else:
            raise ValueError(f"head_fusion desconhecido: {head_fusion}")
        
        # Adicionar conexão residual (skip connection)
        eye = torch.eye(attn.size(-1), device=attn.device)
        attn = attn + eye
        
        # Normalizar
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Rollout: multiplicar atenções de todas as camadas
        joint_attn = attn[0]
        for i in range(1, attn.size(0)):
            joint_attn = attn[i] @ joint_attn
        
        return joint_attn  # (B, N, N)
    
    def generate_attention_map(
        self,
        image_path: str,
        text: str,
        image_size: int = 224,
        head_fusion: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Gera mapa de atenção para um par imagem-texto.
        
        Args:
            image_path: Caminho da imagem.
            text: Texto associado.
            image_size: Tamanho da imagem.
            head_fusion: Método de fusão de heads.
            
        Returns:
            Tupla (imagem original, mapa de atenção, similaridade).
        """
        self.model.model.eval()
        
        # Carregar e preprocessar imagem
        image = Image.open(image_path)
        image = paste_center_on_canvas(image, image_size)
        image_np = np.array(image)
        
        # Preparar inputs
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        # Obter o modelo real (pode ser wrapper ou modelo direto)
        real_model = getattr(self.model, 'model', self.model)
        
        # Garantir que output_attentions está habilitado em todas as configs
        real_model.config.output_attentions = True
        if hasattr(real_model, 'vision_model') and hasattr(real_model.vision_model, 'config'):
            real_model.vision_model.config.output_attentions = True
        if hasattr(real_model, 'text_model') and hasattr(real_model.text_model, 'config'):
            real_model.text_model.config.output_attentions = True
        
        attentions = None
        
        with torch.no_grad():
            # Primeiro, tentar extrair attentions diretamente do vision_model
            # Isso funciona melhor porque passamos output_attentions diretamente
            if hasattr(real_model, 'vision_model'):
                vision_outputs = real_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_attentions=True,
                    return_dict=True
                )
                attentions = vision_outputs.attentions
            
            # Se ainda não temos attentions, tentar pelo modelo completo
            if attentions is None:
                outputs = real_model(
                    **inputs,
                    output_attentions=True
                )
                if hasattr(outputs, 'vision_model_output') and outputs.vision_model_output is not None:
                    attentions = outputs.vision_model_output.attentions
        
        if attentions is None:
            raise ValueError(
                "Modelo não retornou attention maps. "
                "Este modelo pode não suportar visualização de atenção."
            )
        
        # Compute rollout
        joint_attn = self.compute_rollout(attentions, head_fusion)
        
        # Extrair atenção do CLS token para patches
        # joint_attn: (B, N, N) onde N = 1 (CLS) + num_patches
        # Queremos: atenção do CLS (idx 0) para todos os patches (idx 1:)
        mask = joint_attn[0, 0, 1:]  # Remove CLS, fica com patches
        
        # Reshape para grid
        num_patches = mask.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        cam = mask.reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize para tamanho da imagem
        cam = cv2.resize(cam, (image_size, image_size))
        
        # Normalizar
        cam = cam / (cam.max() + 1e-8)
        
        # Calcular similaridade usando os métodos do modelo
        with torch.no_grad():
            image_embeds = real_model.get_image_features(pixel_values=inputs['pixel_values'])
            text_embeds = real_model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask')
            )
        
        # Normalizar embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        similarity = (image_embeds @ text_embeds.t()).item()
        
        return image_np, cam, similarity
    
    def visualize(
        self,
        image_path: str,
        text: str,
        image_size: int = 224,
        save_path: Optional[str] = None,
        show: bool = True,
        head_fusion: str = "mean",
        alpha: float = 0.4
    ) -> Optional[np.ndarray]:
        """
        Visualiza attention rollout para um par imagem-texto.
        
        Args:
            image_path: Caminho da imagem.
            text: Texto associado.
            image_size: Tamanho da imagem.
            save_path: Caminho para salvar figura (opcional).
            show: Se True, exibe figura.
            head_fusion: Método de fusão de heads.
            alpha: Opacidade do overlay.
            
        Returns:
            Overlay da atenção na imagem.
        """
        img_np, cam, similarity = self.generate_attention_map(
            image_path, text, image_size, head_fusion
        )
        
        # Criar heatmap colorido
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
        
        # Plotar
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Imagem Original")
        axes[0].axis("off")
        
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Attention Rollout")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (sim={similarity:.4f})")
        axes[2].axis("off")
        
        plt.suptitle(f"Texto: {text}", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Figura salva em: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return overlay


def visualize_attention(
    model,
    processor,
    image_path: str,
    text: str,
    device: torch.device,
    image_size: int = 224,
    save_path: Optional[str] = None,
    show: bool = True
) -> np.ndarray:
    """
    Função de conveniência para visualizar atenção.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor.
        image_path: Caminho da imagem.
        text: Texto.
        device: Dispositivo.
        image_size: Tamanho da imagem.
        save_path: Caminho para salvar.
        show: Se deve exibir.
        
    Returns:
        Overlay da atenção.
    """
    rollout = AttentionRollout(model, processor, device)
    return rollout.visualize(
        image_path, text, image_size, save_path, show
    )


def generate_attention_map(
    model,
    processor,
    image_path: str,
    text: str,
    device: torch.device,
    image_size: int = 224
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Função de conveniência para gerar mapa de atenção.
    
    Returns:
        Tupla (imagem, attention_map, similaridade).
    """
    rollout = AttentionRollout(model, processor, device)
    return rollout.generate_attention_map(image_path, text, image_size)


def batch_attention_visualization(
    model,
    processor,
    pairs: List[Tuple[str, str]],
    device: torch.device,
    output_dir: str,
    image_size: int = 224,
    max_samples: int = 20
):
    """
    Gera visualizações de atenção para múltiplos pares.
    
    Args:
        model: Wrapper do modelo.
        processor: Processor.
        pairs: Lista de (image_path, text).
        device: Dispositivo.
        output_dir: Diretório de saída.
        image_size: Tamanho da imagem.
        max_samples: Máximo de amostras.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rollout = AttentionRollout(model, processor, device)
    
    for i, (img_path, text) in enumerate(pairs[:max_samples]):
        save_file = output_path / f"attention_{i:03d}.png"
        rollout.visualize(
            img_path, text, image_size,
            save_path=str(save_file),
            show=False
        )
    
    print(f"✅ {min(len(pairs), max_samples)} visualizações salvas em: {output_dir}")

