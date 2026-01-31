"""
Wrappers para modelos CLIP, TinyCLIP e SigLIP.

Cada wrapper implementa a interface BaseMultimodalModel.
"""

from typing import Dict, Any, Optional
import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
    SiglipImageProcessor,
    SiglipTokenizer,
)

from src.models.base import BaseMultimodalModel


class CLIPWrapper(BaseMultimodalModel):
    """
    Wrapper para modelos CLIP padrão do HuggingFace.
    
    Suporta diferentes tamanhos: ViT-B/32, ViT-B/16, ViT-L/14, etc.
    """
    
    def __init__(self):
        super().__init__()
    
    def load_pretrained(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_attentions: bool = True,
        **kwargs
    ) -> None:
        """
        Carrega modelo CLIP pré-treinado.
        
        Args:
            model_name: Nome do modelo no HuggingFace.
            output_attentions: Se True, habilita output de attention maps.
        """
        self.model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs
        )
        self.model.config.output_attentions = output_attentions
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        total, trainable = self.count_parameters()
        print(f"📦 CLIP carregado: {model_name}")
        print(f"   Parâmetros: {total:,} total, {trainable:,} treináveis")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass do CLIP."""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "vision_model_output": outputs.vision_model_output if output_attentions else None,
            "text_model_output": outputs.text_model_output if output_attentions else None,
        }
    
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Obtém features da imagem."""
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        return self._extract_tensor(outputs)
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Obtém features do texto."""
        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self._extract_tensor(outputs)
    
    def _extract_tensor(self, outputs) -> torch.Tensor:
        """Extrai tensor de output que pode ser tensor ou objeto."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]
        if hasattr(outputs, 'image_embeds'):
            return outputs.image_embeds
        if hasattr(outputs, 'text_embeds'):
            return outputs.text_embeds
        raise ValueError(f"Não foi possível extrair tensor de: {type(outputs)}")


class TinyCLIPWrapper(BaseMultimodalModel):
    """
    Wrapper para TinyCLIP - versão compacta do CLIP.
    
    Usa a mesma interface do CLIP padrão.
    """
    
    def __init__(self):
        super().__init__()
    
    def load_pretrained(
        self,
        model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        output_attentions: bool = True,
        **kwargs
    ) -> None:
        """
        Carrega modelo TinyCLIP pré-treinado.
        
        Args:
            model_name: Nome do modelo no HuggingFace.
            output_attentions: Se True, habilita output de attention maps.
        """
        self.model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs
        )
        self.model.config.output_attentions = output_attentions
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        total, trainable = self.count_parameters()
        print(f"📦 TinyCLIP carregado: {model_name}")
        print(f"   Parâmetros: {total:,} total, {trainable:,} treináveis")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass do TinyCLIP."""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "vision_model_output": outputs.vision_model_output if output_attentions else None,
            "text_model_output": outputs.text_model_output if output_attentions else None,
        }
    
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Obtém features da imagem."""
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        return self._extract_tensor(outputs)
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Obtém features do texto."""
        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self._extract_tensor(outputs)
    
    def _extract_tensor(self, outputs) -> torch.Tensor:
        """Extrai tensor de output que pode ser tensor ou objeto."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]
        if hasattr(outputs, 'image_embeds'):
            return outputs.image_embeds
        if hasattr(outputs, 'text_embeds'):
            return outputs.text_embeds
        raise ValueError(f"Não foi possível extrair tensor de: {type(outputs)}")


class SigLIPWrapper(BaseMultimodalModel):
    """
    Wrapper para SigLIP - CLIP com Sigmoid Loss.
    
    SigLIP usa sigmoid loss ao invés de softmax/InfoNCE,
    permitindo treinamento mais eficiente em batches grandes.
    """
    
    def __init__(self):
        super().__init__()
    
    def load_pretrained(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        output_attentions: bool = True,
        **kwargs
    ) -> None:
        """
        Carrega modelo SigLIP pré-treinado.
        
        Args:
            model_name: Nome do modelo no HuggingFace.
            output_attentions: Se True, habilita output de attention maps.
        """
        self.model = SiglipModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs
        )
        self.model.config.output_attentions = output_attentions
        
        # Carregar componentes separadamente para evitar bug no AutoTokenizer
        image_processor = SiglipImageProcessor.from_pretrained(model_name)
        tokenizer = SiglipTokenizer.from_pretrained(model_name)
        self.processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
        
        total, trainable = self.count_parameters()
        print(f"📦 SigLIP carregado: {model_name}")
        print(f"   Parâmetros: {total:,} total, {trainable:,} treináveis")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass do SigLIP.
        
        SigLIP usa uma estrutura ligeiramente diferente do CLIP padrão.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # SigLIP retorna logits_per_image e logits_per_text diretamente
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "vision_model_output": getattr(outputs, 'vision_model_output', None),
            "text_model_output": getattr(outputs, 'text_model_output', None),
        }
    
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Obtém features da imagem."""
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        return self._extract_tensor(outputs)
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Obtém features do texto."""
        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self._extract_tensor(outputs)
    
    def _extract_tensor(self, outputs) -> torch.Tensor:
        """Extrai tensor de output que pode ser tensor ou objeto."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]
        if hasattr(outputs, 'image_embeds'):
            return outputs.image_embeds
        if hasattr(outputs, 'text_embeds'):
            return outputs.text_embeds
        raise ValueError(f"Não foi possível extrair tensor de: {type(outputs)}")


def load_from_checkpoint(
    checkpoint_path: str,
    model_type: str = "clip",
    output_attentions: bool = True
) -> BaseMultimodalModel:
    """
    Carrega modelo de um checkpoint salvo.
    
    Args:
        checkpoint_path: Caminho para o checkpoint.
        model_type: Tipo do modelo ("clip", "tinyclip", "siglip").
        output_attentions: Se True, habilita attention outputs.
        
    Returns:
        Wrapper do modelo carregado.
    """
    if model_type in ["clip", "tinyclip"]:
        wrapper = TinyCLIPWrapper() if model_type == "tinyclip" else CLIPWrapper()
        wrapper.model = CLIPModel.from_pretrained(
            checkpoint_path,
            attn_implementation="eager"
        )
        wrapper.model.config.output_attentions = output_attentions
        wrapper.processor = CLIPProcessor.from_pretrained(checkpoint_path)
    elif model_type == "siglip":
        wrapper = SigLIPWrapper()
        wrapper.model = SiglipModel.from_pretrained(
            checkpoint_path,
            attn_implementation="eager"
        )
        wrapper.model.config.output_attentions = output_attentions
        # Carregar componentes separadamente para evitar bug no AutoTokenizer
        image_processor = SiglipImageProcessor.from_pretrained(checkpoint_path)
        tokenizer = SiglipTokenizer.from_pretrained(checkpoint_path)
        wrapper.processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
    
    print(f"📂 Modelo carregado de: {checkpoint_path}")
    return wrapper

