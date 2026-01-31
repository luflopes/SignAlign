"""
Visualizações avançadas para análise de modelos.

Inclui matriz de similaridade gráfica e comparações visuais.
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import seaborn as sns

from src.data.dataset import paste_center_on_canvas


class SimilarityMatrixVisualizer:
    """
    Visualizador de matriz de similaridade texto-imagem.
    
    Exibe uma matriz com thumbnails de imagens e textos,
    mostrando os valores de similaridade de forma gráfica.
    """
    
    def __init__(
        self,
        predictor,
        image_size: int = 224,
        thumbnail_size: int = 80
    ):
        """
        Args:
            predictor: SignaturePredictor carregado.
            image_size: Tamanho das imagens para processamento.
            thumbnail_size: Tamanho dos thumbnails na visualização.
        """
        self.predictor = predictor
        self.image_size = image_size
        self.thumbnail_size = thumbnail_size
    
    def create_thumbnail(self, image_path: str) -> np.ndarray:
        """Cria thumbnail de uma imagem."""
        img = Image.open(image_path)
        img = paste_center_on_canvas(img, self.image_size)
        img = img.resize((self.thumbnail_size, self.thumbnail_size), Image.LANCZOS)
        return np.array(img)
    
    def plot_similarity_matrix(
        self,
        texts: List[str],
        images: List[str],
        title: str = "Matriz de Similaridade Texto × Imagem",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = None,
        annotate: bool = True,
        cmap: str = "RdYlGn",
        highlight_diagonal: bool = True
    ) -> np.ndarray:
        """
        Plota matriz de similaridade com thumbnails.
        
        Args:
            texts: Lista de textos (nomes).
            images: Lista de caminhos de imagens.
            title: Título do gráfico.
            save_path: Caminho para salvar.
            show: Se deve exibir.
            figsize: Tamanho da figura.
            annotate: Se deve anotar valores.
            cmap: Colormap.
            highlight_diagonal: Se deve destacar diagonal.
            
        Returns:
            Matriz de similaridade.
        """
        # Calcular similaridades
        similarity_matrix = self.predictor.batch_similarity(texts, images)
        
        n_texts = len(texts)
        n_images = len(images)
        
        # Calcular tamanho da figura
        if figsize is None:
            base_size = 2
            figsize = (
                base_size * n_images + 3,
                base_size * n_texts + 2
            )
        
        # Criar figura com GridSpec
        fig = plt.figure(figsize=figsize)
        
        # Layout: thumbnails de imagens no topo, textos na esquerda, matriz no centro
        gs = fig.add_gridspec(
            n_texts + 1, n_images + 1,
            width_ratios=[1] + [1] * n_images,
            height_ratios=[1] + [1] * n_texts,
            wspace=0.05, hspace=0.05
        )
        
        # Thumbnails das imagens (topo)
        for j, img_path in enumerate(images):
            ax = fig.add_subplot(gs[0, j + 1])
            thumbnail = self.create_thumbnail(img_path)
            ax.imshow(thumbnail)
            ax.axis('off')
        
        # Textos (esquerda) e células da matriz
        for i, text in enumerate(texts):
            # Texto
            ax_text = fig.add_subplot(gs[i + 1, 0])
            ax_text.text(
                0.95, 0.5, text,
                ha='right', va='center',
                fontsize=9, fontweight='bold',
                transform=ax_text.transAxes
            )
            ax_text.axis('off')
            
            # Células da matriz
            for j in range(n_images):
                ax = fig.add_subplot(gs[i + 1, j + 1])
                
                similarity = similarity_matrix[i, j]
                
                # Cor baseada na similaridade
                norm_sim = (similarity + 1) / 2  # Normalizar para [0, 1]
                color = plt.cm.get_cmap(cmap)(norm_sim)
                
                ax.set_facecolor(color)
                
                # Destacar diagonal
                if highlight_diagonal and i == j:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(3)
                
                # Anotar valor
                if annotate:
                    text_color = 'white' if norm_sim < 0.5 else 'black'
                    ax.text(
                        0.5, 0.5, f'{similarity:.3f}',
                        ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color=text_color,
                        transform=ax.transAxes
                    )
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=-1, vmax=1)
        )
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Similaridade', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Matriz salva em: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return similarity_matrix
    
    def plot_heatmap(
        self,
        texts: List[str],
        images: List[str],
        title: str = "Heatmap de Similaridade",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "RdYlGn"
    ) -> np.ndarray:
        """
        Plota heatmap simples de similaridade.
        
        Versão mais compacta para muitos itens.
        """
        similarity_matrix = self.predictor.batch_similarity(texts, images)
        
        plt.figure(figsize=figsize)
        
        # Criar labels truncados para imagens
        img_labels = [Path(p).stem[:15] for p in images]
        text_labels = [t[:20] for t in texts]
        
        sns.heatmap(
            similarity_matrix,
            xticklabels=img_labels,
            yticklabels=text_labels,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Similaridade'}
        )
        
        plt.xlabel('Imagens', fontsize=12)
        plt.ylabel('Textos', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Heatmap salvo em: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return similarity_matrix


def plot_similarity_matrix(
    predictor,
    texts: List[str],
    images: List[str],
    **kwargs
) -> np.ndarray:
    """
    Função de conveniência para plotar matriz de similaridade.
    """
    visualizer = SimilarityMatrixVisualizer(predictor)
    return visualizer.plot_similarity_matrix(texts, images, **kwargs)


def plot_retrieval_results(
    predictor,
    query_text: str,
    candidate_images: List[str],
    top_k: int = 5,
    correct_idx: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = None
):
    """
    Visualiza resultados de retrieval para uma query.
    
    Args:
        predictor: SignaturePredictor.
        query_text: Texto da query.
        candidate_images: Imagens candidatas.
        top_k: Número de resultados a mostrar.
        correct_idx: Índice da imagem correta (para destacar).
        title: Título personalizado.
        save_path: Caminho para salvar.
        show: Se deve exibir.
        figsize: Tamanho da figura.
    """
    # Rankear candidatas
    ranked = predictor.rank_signatures(query_text, candidate_images)[:top_k]
    
    if figsize is None:
        figsize = (4 * top_k, 5)
    
    fig, axes = plt.subplots(1, top_k, figsize=figsize)
    if top_k == 1:
        axes = [axes]
    
    # Encontrar índice correto no ranking
    correct_img = candidate_images[correct_idx] if correct_idx is not None else None
    
    for i, (img_path, similarity) in enumerate(ranked):
        ax = axes[i]
        
        # Carregar imagem
        img = Image.open(img_path)
        img = paste_center_on_canvas(img, 224)
        
        ax.imshow(img)
        
        # Destacar se for o correto
        is_correct = img_path == correct_img
        border_color = 'green' if is_correct else ('red' if correct_img else 'gray')
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_correct else 1)
        
        # Título com rank e similaridade
        status = "✓ CORRETO" if is_correct else ""
        ax.set_title(f"Rank {i+1}\nSim: {similarity:.4f}\n{status}", fontsize=10)
        ax.axis('off')
    
    fig_title = title or f"Retrieval: '{query_text}'"
    plt.suptitle(fig_title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_grid(
    predictor,
    pairs: List[Tuple[str, str]],
    num_cols: int = 4,
    title: str = "Comparação de Pares Texto-Assinatura",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plota grade de comparações texto-assinatura.
    
    Args:
        predictor: SignaturePredictor.
        pairs: Lista de (texto, caminho_imagem).
        num_cols: Número de colunas na grade.
        title: Título.
        save_path: Caminho para salvar.
        show: Se deve exibir.
    """
    num_pairs = len(pairs)
    num_rows = (num_pairs + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = np.array(axes).reshape(-1)
    
    for i, (text, img_path) in enumerate(pairs):
        ax = axes[i]
        
        # Calcular similaridade
        similarity = predictor.compute_similarity(text, img_path)
        
        # Carregar imagem
        img = Image.open(img_path)
        img = paste_center_on_canvas(img, 224)
        
        ax.imshow(img)
        ax.set_title(f"{text}\nSim: {similarity:.4f}", fontsize=9)
        ax.axis('off')
        
        # Cor da borda baseada na similaridade
        color = 'green' if similarity > 0.2 else ('orange' if similarity > 0.1 else 'red')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Esconder eixos vazios
    for i in range(num_pairs, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

