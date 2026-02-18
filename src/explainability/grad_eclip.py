import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradECLIP:
    """
    Grad-ECLIP: Gradient-based Visual and Textual Explanations for CLIP.
    
    Adaptado para modelos HuggingFace (CLIP, TinyCLIP, SigLIP).
    Baseado em: https://github.com/Cyang-Zhao/Grad-Eclip
    Referência: arxiv.org/html/2502.18816v1
    """
    
    def __init__(self, model_wrapper, device="cpu"):
        """
        Args:
            model_wrapper: Wrapper do modelo (CLIPWrapper, TinyCLIPWrapper, SigLIPWrapper)
            device: Dispositivo para computação
        """
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.processor = model_wrapper.processor
        self.device = device
        
        # Identificar tipo de modelo
        self.is_siglip = hasattr(self.model, 'vision_model') and 'siglip' in str(type(self.model)).lower()
        
        # Obter informações do modelo
        self.vision_model = self.model.vision_model
        self.patch_size = self._get_patch_size()
        self.num_heads = self.vision_model.config.num_attention_heads
        self.hidden_size = self.vision_model.config.hidden_size
        
        # Storage para hooks
        self.hooks = []
        self.stored_data = {}
    
    def _get_patch_size(self):
        """Obtém tamanho do patch do modelo."""
        if hasattr(self.vision_model, 'embeddings'):
            if hasattr(self.vision_model.embeddings, 'patch_embedding'):
                conv = self.vision_model.embeddings.patch_embedding
            elif hasattr(self.vision_model.embeddings, 'patch_embeddings'):
                conv = self.vision_model.embeddings.patch_embeddings.projection
            else:
                return 16
            return conv.kernel_size[0] if hasattr(conv, 'kernel_size') else 16
        return 16
    
    def _register_hooks(self, layer_idx=-1):
        """Registra hooks para capturar q, k, v da camada especificada."""
        self._clear_hooks()
        
        # Obter camadas do encoder
        if hasattr(self.vision_model, 'encoder'):
            layers = self.vision_model.encoder.layers
        else:
            raise ValueError("Modelo não tem encoder layers")
        
        target_layer = layers[layer_idx]
        attn = target_layer.self_attn
        
        # Hook para capturar q, k, v antes da atenção
        def make_qkv_hook(name):
            def hook(module, input, output):
                self.stored_data[name] = output.detach().clone()
                self.stored_data[name].requires_grad_(True)
            return hook
        
        # Registrar hooks nas projeções
        self.hooks.append(attn.q_proj.register_forward_hook(make_qkv_hook('q')))
        self.hooks.append(attn.k_proj.register_forward_hook(make_qkv_hook('k')))
        self.hooks.append(attn.v_proj.register_forward_hook(make_qkv_hook('v')))
        
        # Hook para o output da atenção
        def attn_output_hook(module, input, output):
            # output pode ser tuple (attn_output, attn_weights) ou tensor
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output
            self.stored_data['attn_output'] = attn_out
        
        self.hooks.append(attn.register_forward_hook(attn_output_hook))
    
    def _clear_hooks(self):
        """Remove todos os hooks registrados."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.stored_data = {}
    
    def _sim_qk(self, q, k, num_patches):
        """
        Calcula peso espacial baseado na similaridade q-k.
        Mede quão relevante cada patch é para o token CLS.
        """
        # q: (batch, seq_len, hidden_size), k: (batch, seq_len, hidden_size)
        # CLS token é o primeiro
        q_cls = F.normalize(q[:, :1, :], dim=-1)  # (batch, 1, hidden)
        k_patches = F.normalize(k[:, 1:, :], dim=-1)  # (batch, num_patches, hidden)
        
        # Similaridade coseno entre CLS query e patches keys
        cosine_qk = (q_cls * k_patches).sum(-1)  # (batch, num_patches)
        
        # Normalizar para [0, 1]
        cosine_qk = cosine_qk - cosine_qk.min(dim=-1, keepdim=True)[0]
        cosine_qk = cosine_qk / (cosine_qk.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return cosine_qk.squeeze(0)  # (num_patches,)
    
    def compute_grad_eclip(self, image, text, n_layers=1):
        """
        Computa o mapa de explicação Grad-ECLIP.
        
        Args:
            image: PIL Image
            text: String do texto
            n_layers: Número de camadas finais para usar
            
        Returns:
            attention_map: numpy array do mapa de atenção
            similarity: float da similaridade imagem-texto
        """
        self.model.eval()
        
        # Processar inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Habilitar gradientes
        pixel_values = inputs['pixel_values'].clone().requires_grad_(True)
        
        # Calcular tamanho do feature map
        img_size = pixel_values.shape[-1]  # Assume square
        feat_h = feat_w = img_size // self.patch_size
        num_patches = feat_h * feat_w
        
        all_maps = []
        
        for layer_idx in range(-n_layers, 0):
            self._register_hooks(layer_idx)
            
            # Forward pass - precisamos que retenha o grafo computacional
            with torch.enable_grad():
                # Processar imagem pelo vision model
                vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Obter embeddings
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    image_embeds = vision_outputs.pooler_output
                else:
                    image_embeds = vision_outputs.last_hidden_state[:, 0, :]
                
                # Projetar para espaço de embedding final
                if hasattr(self.model, 'visual_projection'):
                    image_embeds = self.model.visual_projection(image_embeds)
                
                # Processar texto
                text_inputs = {k: v for k, v in inputs.items() if k != 'pixel_values'}
                text_outputs = self.model.text_model(**text_inputs)
                
                if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                    text_embeds = text_outputs.pooler_output
                else:
                    text_embeds = text_outputs.last_hidden_state[:, -1, :]  # EOS token para CLIP
                
                if hasattr(self.model, 'text_projection'):
                    text_embeds = self.model.text_projection(text_embeds)
                
                # Normalizar
                image_embeds = F.normalize(image_embeds, dim=-1)
                text_embeds = F.normalize(text_embeds, dim=-1)
                
                # Similaridade
                similarity = (image_embeds * text_embeds).sum(-1)
            
            # Obter q, k, v capturados
            q = self.stored_data.get('q')
            k = self.stored_data.get('k')
            v = self.stored_data.get('v')
            attn_output = self.stored_data.get('attn_output')
            
            if q is None or k is None or v is None:
                print(f"⚠️ Não foi possível capturar q/k/v da camada {layer_idx}")
                continue
            
            # Reshape q, k, v: (seq_len, batch, hidden) -> (batch, seq_len, hidden)
            if q.dim() == 3 and q.shape[0] != 1:
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)
            
            # Calcular peso espacial sim_qk
            spatial_weight = self._sim_qk(q, k, num_patches)
            
            # Gradiente da similaridade em relação ao attn_output
            if attn_output is not None and attn_output.requires_grad:
                grad = torch.autograd.grad(
                    similarity,
                    attn_output,
                    retain_graph=True,
                    create_graph=False
                )[0]
            else:
                # Fallback: usar gradiente em relação a v
                v_for_grad = v.clone().requires_grad_(True)
                grad = torch.autograd.grad(
                    similarity,
                    pixel_values,
                    retain_graph=True,
                    create_graph=False
                )[0]
                # Simplificar para spatial attention
                grad = grad.mean(dim=1).flatten(1)  # (batch, H*W)
                grad = grad[:, :num_patches]
            
            # Reshape grad se necessário
            if grad.dim() == 3:
                if grad.shape[0] != 1:
                    grad = grad.transpose(0, 1)  # (batch, seq_len, hidden)
                grad_cls = grad[:, :1, :]  # (batch, 1, hidden)
            else:
                grad_cls = grad.unsqueeze(1)
            
            # Valor dos patches (excluindo CLS)
            v_patches = v[:, 1:, :]  # (batch, num_patches, hidden)
            
            # Grad-ECLIP: gradiente * valores * peso espacial
            emap = (grad_cls * v_patches * spatial_weight.unsqueeze(0).unsqueeze(-1)).sum(-1)
            emap = F.relu(emap)  # Apenas contribuições positivas
            
            all_maps.append(emap.squeeze())
            
            self._clear_hooks()
        
        # Combinar mapas de todas as camadas
        if len(all_maps) > 0:
            combined_map = torch.stack(all_maps, dim=0).sum(0)
        else:
            # Fallback: usar gradient direto
            grad_fallback = torch.autograd.grad(similarity, pixel_values, retain_graph=True)[0]
            combined_map = grad_fallback.abs().mean(dim=1).flatten()[:num_patches]
        
        # Reshape para grid espacial
        attention_map = combined_map.reshape(feat_h, feat_w)
        
        # Normalizar
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / (attention_map.max() + 1e-8)
        
        return attention_map.detach().cpu().numpy(), similarity.item()
    
    def _paste_center_on_canvas(self, img, canvas_size=224, background=(255, 255, 255)):
        """Centraliza imagem em canvas quadrado (igual ao treinamento)."""
        img = img.convert("RGBA")
        w, h = img.size
        scale = min(canvas_size / w, canvas_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (canvas_size, canvas_size), color=background + (255,))
        offset = ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2)
        canvas.paste(img_resized, offset, img_resized)
        return canvas.convert("RGB")
    
    def visualize(self, image_path, text, save_path=None, show=True, alpha=0.5):
        """
        Visualiza o mapa Grad-ECLIP sobreposto na imagem.
        
        Args:
            image_path: Caminho para a imagem
            text: Texto para comparar
            save_path: Caminho para salvar a visualização
            show: Se True, mostra a figura
            alpha: Transparência do heatmap
            
        Returns:
            dict com attention_map, similarity, e visualização
        """
        # Aplicar mesmo pré-processamento do treinamento
        image = self._paste_center_on_canvas(Image.open(image_path))
        canvas_size = 224  # Tamanho do canvas usado no treinamento
        
        # Computar Grad-ECLIP
        attention_map, similarity = self.compute_grad_eclip(image, text, n_layers=1)
        
        # Resize do mapa para o tamanho do canvas (224x224)
        # O attention map sai com shape ~(14,14), resize para 224x224
        resize = T.Resize((canvas_size, canvas_size), interpolation=T.InterpolationMode.BILINEAR)
        attention_map_resized = resize(
            torch.from_numpy(attention_map).unsqueeze(0)
        )[0].numpy()
        
        # Criar heatmap colorido
        heatmap = cv2.applyColorMap(
            (attention_map_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Sobrepor na imagem processada (canvas 224x224)
        image_np = np.array(image)
        overlay = np.clip(
            image_np * (1 - alpha) + heatmap * alpha, 
            0, 255
        ).astype(np.uint8)
        
        if show:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Imagem processada (como o modelo vê)
            axes[0].imshow(image)
            axes[0].set_title(f"Imagem (canvas 224x224)\nTexto: {text[:30]}...")
            axes[0].axis('off')
            
            # Mapa de atenção
            axes[1].imshow(attention_map_resized, cmap='jet')
            axes[1].set_title(f"Grad-ECLIP Map\nSimilaridade: {similarity:.4f}")
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (atenção sobre canvas)")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 Salvo em: {save_path}")
            
            plt.show()
        
        return {
            "attention_map": attention_map,
            "attention_map_resized": attention_map_resized,
            "similarity": similarity,
            "overlay": overlay
        }