"""
Trainer principal para experimentos SignAlign.

Gerencia o loop de treinamento, validação, checkpointing e logging.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from src.config.experiment_config import ExperimentConfig
from src.models.base import BaseMultimodalModel
from src.models.registry import create_model
from src.losses.combined import CombinedLoss, create_loss_function
from src.data.dataset import SignatureNameDataset, collate_fn
from src.data.augmentations import get_train_transform, get_val_transform
from src.data.batch_builder import (
    load_dataset_pairs,
    create_train_val_split,
    build_unique_name_batches,
    create_fixed_evaluation_data,
)
from src.training.scheduler import create_scheduler
from src.evaluation.metrics import compute_all_metrics
from src.utils.logging import ExperimentLogger
from src.utils.seed import set_seed, get_device


class Trainer:
    """
    Trainer para experimentos de alinhamento texto-assinatura.
    
    Gerencia todo o ciclo de treinamento incluindo:
    - Preparação de dados
    - Loop de treinamento
    - Validação periódica
    - Checkpointing do melhor modelo
    - Logging estruturado
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração completa do experimento.
        """
        self.config = config
        self.device = get_device()
        
        # Definir seed para reprodutibilidade
        set_seed(config.seed)
        
        # Inicializar componentes
        self.model: Optional[BaseMultimodalModel] = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn: Optional[CombinedLoss] = None
        self.logger: Optional[ExperimentLogger] = None
        
        # Dados
        self.train_pairs: List[Tuple[str, str]] = []
        self.val_pairs: List[Tuple[str, str]] = []
        self.fixed_eval_data: Dict = {}
        
        # Estado do treinamento
        self.current_epoch = 0
        self.best_metric = -1.0  # Começar negativo para aceitar qualquer melhoria
        self.best_epoch = 1  # Default para primeira época
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
    
    def setup(
        self, 
        pretrained_checkpoint: Optional[str] = None,
        val_csv: Optional[str] = None
    ) -> None:
        """
        Configura todos os componentes para treinamento.
        
        Args:
            pretrained_checkpoint: Caminho para checkpoint pré-treinado a ser carregado.
            val_csv: Caminho para CSV de validação separado.
        """
        print("=" * 60)
        print(f"🚀 Configurando experimento: {self.config.name}")
        print("=" * 60)
        
        # 1. Logger
        self.logger = ExperimentLogger(
            experiment_name=self.config.name,
            output_dir=self.config.output_dir
        )
        self.logger.log_config(self.config.to_dict())
        
        # 2. Modelo
        print("\n📦 Carregando modelo...")
        self.model = create_model(self.config.model, self.device)
        self.processor = self.model.get_processor()
        
        # 2.1 Carregar checkpoint pré-treinado se fornecido
        if pretrained_checkpoint:
            self._load_pretrained_checkpoint(pretrained_checkpoint)
        
        # 3. Loss
        print("\n🎯 Configurando loss...")
        self.loss_fn = create_loss_function(self.config.loss)
        
        # 4. Otimizador
        print("\n⚙️ Configurando otimizador...")
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # 5. Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.training.scheduler,
            self.config.training.epochs
        )
        
        # 6. Dados
        print("\n📂 Carregando dados...")
        self._setup_data(val_csv=val_csv)
        
        print("\n✅ Setup completo!")
        print("=" * 60)
    
    def _load_pretrained_checkpoint(self, checkpoint_path: str) -> None:
        """
        Carrega pesos de um checkpoint pré-treinado.
        
        Args:
            checkpoint_path: Caminho para o diretório do checkpoint.
        """
        from transformers import CLIPModel, CLIPProcessor
        
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
        
        print(f"   📥 Carregando checkpoint: {checkpoint_path}")
        
        # Carregar modelo do checkpoint
        loaded_model = CLIPModel.from_pretrained(checkpoint_dir)
        
        # Copiar pesos para o modelo atual
        self.model.model.load_state_dict(loaded_model.state_dict())
        
        # Carregar processor também (por consistência)
        self.processor = CLIPProcessor.from_pretrained(checkpoint_dir)
        
        print(f"   ✅ Checkpoint carregado com sucesso!")
    
    def _setup_data(self, val_csv: Optional[str] = None) -> None:
        """
        Prepara datasets de treino e validação.
        
        Args:
            val_csv: Caminho para CSV de validação separado (opcional).
                     Se fornecido, não faz split e usa esse arquivo para validação.
        """
        if val_csv:
            # Usar CSVs separados para treino e validação
            print(f"   📄 Usando CSV de validação separado: {val_csv}")
            
            # Treino
            self.train_pairs = load_dataset_pairs(
                self.config.data.dataset_csv,
                exclude_unknown=self.config.data.exclude_unknown,
                images_base_path=self.config.data.images_base_path,
                max_samples=self.config.data.max_samples
            )
            
            # Validação
            self.val_pairs = load_dataset_pairs(
                val_csv,
                exclude_unknown=self.config.data.exclude_unknown,
                images_base_path=self.config.data.images_base_path,
                max_samples=None  # Não limitar validação
            )
        else:
            # Carregar pares e fazer split
            all_pairs = load_dataset_pairs(
                self.config.data.dataset_csv,
                exclude_unknown=self.config.data.exclude_unknown,
                images_base_path=self.config.data.images_base_path,
                max_samples=self.config.data.max_samples
            )
            
            # Split treino/validação
            self.train_pairs, self.val_pairs = create_train_val_split(
                all_pairs,
                val_ratio=self.config.data.val_ratio,
                seed=self.config.seed
            )
        
        print(f"📊 Split de dados:")
        print(f"   - Treino: {len(self.train_pairs)} amostras")
        print(f"   - Validação: {len(self.val_pairs)} amostras")
        
        # Aviso se poucos dados de validação
        if len(self.val_pairs) < 10:
            print(f"⚠️ AVISO: Apenas {len(self.val_pairs)} amostras de validação. Métricas podem ser imprecisas.")
        
        # Criar dados fixos de avaliação
        self.fixed_eval_data = create_fixed_evaluation_data(
            self.val_pairs,
            max_negative_samples=max(self.config.evaluation.negative_samples),
            seed=self.config.seed
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Executa o loop de treinamento completo.
        
        Returns:
            Dicionário com métricas finais e histórico.
        """
        print("\n" + "=" * 60)
        print("🏋️ Iniciando treinamento...")
        print("=" * 60)
        
        # Transforms
        train_transform = get_train_transform(self.config.data.augmentation)
        val_transform = get_val_transform()
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda') if (
            self.config.training.use_amp and self.device.type == "cuda"
        ) else None
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch + 1
            
            # Treinar uma época
            train_metrics = self._train_epoch(train_transform, scaler)
            
            # Validar
            val_metrics = self._validate(val_transform)
            
            # Calcular estatísticas de similaridade no conjunto de validação
            similarity_stats = self._compute_similarity_statistics(val_transform)
            val_metrics.update(similarity_stats)
            
            # Log de época
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_epoch(
                epoch=self.current_epoch,
                train_loss=train_metrics["loss"],
                val_metrics=val_metrics,
                lr=current_lr
            )
            
            # Atualizar scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Para ReduceLROnPlateau, usar métrica de validação
                    metric_for_scheduler = val_metrics.get("accuracy_3_neg", 0.0)
                    self.scheduler.step(metric_for_scheduler)
                else:
                    self.scheduler.step()
            
            # Verificar melhor modelo (tentar diferentes métricas se 3_neg não disponível)
            main_metric = val_metrics.get("accuracy_3_neg") or val_metrics.get("accuracy_2_neg") or val_metrics.get("accuracy_1_neg") or 0.0
            is_best = main_metric > self.best_metric
            
            if is_best:
                self.best_metric = main_metric
                self.best_epoch = self.current_epoch
                self.logger.log_best_model(
                    epoch=self.current_epoch,
                    metric_name="accuracy_3_neg",
                    metric_value=main_metric
                )
            
            # Salvar checkpoint
            should_save = False
            if self.config.training.save_only_best:
                # Só salva se for o melhor modelo
                should_save = is_best
            else:
                # Salva a cada N épocas
                should_save = self.current_epoch % self.config.training.save_every == 0
            
            if should_save:
                self.logger.save_checkpoint(
                    model=self.model.model,
                    processor=self.processor,
                    epoch=self.current_epoch,
                    metric_value=main_metric,
                    is_best=is_best
                )
            
            # Print resumo da época
            print(f"\n✅ Época {self.current_epoch}/{self.config.training.epochs}")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Accuracy: {main_metric:.4f}")
            print(f"   Sim+ {similarity_stats.get('mean_pos_similarity', 0):.4f} | Sim- {similarity_stats.get('mean_neg_similarity', 0):.4f} | Gap: {similarity_stats.get('similarity_gap', 0):.4f}")
            print(f"   LR: {current_lr:.2e}")
            if is_best and main_metric > 0:
                print(f"   🌟 Nova melhor accuracy!")
            
            # Early stopping
            if self.config.training.early_stopping_patience:
                epochs_without_improvement = self.current_epoch - self.best_epoch
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    print(f"\n⚠️ Early stopping após {epochs_without_improvement} épocas sem melhoria.")
                    break
        
        # Finalizar
        self.logger.finalize()
        
        print("\n" + "=" * 60)
        print("🏁 Treinamento finalizado!")
        print(f"   Melhor época: {self.best_epoch}")
        print(f"   Melhor accuracy: {self.best_metric:.4f}")
        print("=" * 60)
        
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
    
    def _train_epoch(
        self,
        transform,
        scaler: Optional[torch.amp.GradScaler]
    ) -> Dict[str, float]:
        """Treina uma época completa."""
        self.model.model.train()
        
        # Construir batches únicos
        unique_batches = build_unique_name_batches(
            self.train_pairs,
            batch_size=self.config.data.batch_size
        )
        random.shuffle(unique_batches)
        
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0
        num_batches = 0
        
        batch_iterator = tqdm(
            unique_batches,
            desc=f"Época {self.current_epoch} - Treino",
            leave=False
        )
        
        for batch_pairs in batch_iterator:
            # Preparar batch
            dataset_batch = SignatureNameDataset(
                batch_pairs, transform, self.config.model.image_size
            )
            batch_data = [dataset_batch[i] for i in range(len(dataset_batch))]
            batch = collate_fn(batch_data, self.processor).to(self.device)
            
            # Forward com AMP
            use_amp = scaler is not None
            with torch.amp.autocast('cuda', enabled=use_amp):
                self.optimizer.zero_grad()
                
                # Forward do modelo
                outputs = self.model.forward(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                )
                
                # Calcular loss
                loss_result = self.loss_fn(
                    image_embeds=outputs['image_embeds'],
                    text_embeds=outputs['text_embeds'],
                    logits_per_image=outputs['logits_per_image'],
                    logits_per_text=outputs['logits_per_text'],
                    return_components=True
                )
                
                loss = loss_result['loss']
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.config.training.max_grad_norm
                )
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Extrair similaridades para logging
            if 'pos_similarities' in loss_result and loss_result['pos_similarities'] is not None:
                total_pos_sim += loss_result['pos_similarities'].mean().item()
            if 'neg_similarities' in loss_result and loss_result['neg_similarities'] is not None:
                total_neg_sim += loss_result['neg_similarities'].mean().item()
            
            batch_iterator.set_postfix(loss=loss.item())
        
        avg_pos_sim = total_pos_sim / num_batches if num_batches > 0 and total_pos_sim > 0 else 0
        avg_neg_sim = total_neg_sim / num_batches if num_batches > 0 and total_neg_sim > 0 else 0
        
        metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else 0,
            "mean_pos_similarity": avg_pos_sim,
            "mean_neg_similarity": avg_neg_sim,
            "similarity_gap": avg_pos_sim - avg_neg_sim if avg_pos_sim > 0 else 0,
        }
        
        self.train_history.append(metrics)
        return metrics
    
    def _validate(self, transform) -> Dict[str, float]:
        """Executa validação completa."""
        self.model.model.eval()
        
        # Calcular métricas para diferentes números de negativos
        all_metrics = {}
        
        for num_neg in self.config.evaluation.negative_samples:
            metrics = compute_all_metrics(
                model=self.model,
                processor=self.processor,
                fixed_eval_data=self.fixed_eval_data,
                num_negative_samples=num_neg,
                device=self.device,
                transform=transform,
                image_size=self.config.model.image_size,
                config=self.config.evaluation
            )
            
            # Prefixar com número de negativos
            for key, value in metrics.items():
                all_metrics[f"{key}_{num_neg}_neg"] = value
        
        self.val_history.append(all_metrics)
        return all_metrics
    
    def evaluate_only(self) -> Dict[str, Any]:
        """
        Executa apenas avaliação (sem treino).
        
        Útil para modelos frozen ou para avaliar checkpoints.
        
        Returns:
            Dicionário com métricas de avaliação.
        """
        print("\n" + "=" * 60)
        print("📊 Executando avaliação (sem treino)...")
        print("=" * 60)
        
        val_transform = get_val_transform()
        
        # Validar
        val_metrics = self._validate(val_transform)
        
        # Calcular similaridades médias no conjunto de validação
        similarity_stats = self._compute_similarity_statistics(val_transform)
        
        # Combinar métricas
        all_metrics = {**val_metrics, **similarity_stats}
        
        # Log
        self.logger.log_epoch(
            epoch=0,
            train_loss=0.0,
            val_metrics=all_metrics,
            lr=0.0
        )
        
        # Finalizar
        self.logger.finalize()
        
        # Print resumo
        print("\n📈 Métricas de Avaliação:")
        for key, value in sorted(all_metrics.items()):
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
        
        print("\n" + "=" * 60)
        print("✅ Avaliação concluída!")
        print("=" * 60)
        
        return {
            "best_epoch": 0,
            "best_metric": val_metrics.get("accuracy_3_neg", 0.0),
            "val_metrics": all_metrics,
            "train_history": [],
            "val_history": [all_metrics],
        }
    
    def _compute_similarity_statistics(self, transform) -> Dict[str, float]:
        """
        Calcula estatísticas de similaridade no conjunto de validação.
        
        Returns:
            Dicionário com similaridades médias positivas e negativas.
        """
        self.model.model.eval()
        
        all_pos_sims = []
        all_neg_sims = []
        
        # Processar dados de validação em batches
        from src.data.dataset import SignatureNameDataset, collate_fn
        
        dataset = SignatureNameDataset(
            self.val_pairs, transform, self.config.model.image_size
        )
        
        batch_size = min(self.config.data.batch_size, len(dataset))
        
        for i in range(0, len(dataset), batch_size):
            batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            batch = collate_fn(batch_data, self.processor).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.forward(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                )
                
                # Calcular similaridades
                image_embeds = outputs['image_embeds']
                text_embeds = outputs['text_embeds']
                
                # Normalizar
                import torch.nn.functional as F
                image_embeds = F.normalize(image_embeds, dim=-1)
                text_embeds = F.normalize(text_embeds, dim=-1)
                
                # Matriz de similaridade
                sims = image_embeds @ text_embeds.t()
                
                # Positivos (diagonal)
                pos_sims = sims.diagonal()
                all_pos_sims.extend(pos_sims.cpu().tolist())
                
                # Negativos (fora da diagonal)
                batch_size_actual = sims.size(0)
                mask = torch.eye(batch_size_actual, device=sims.device).bool()
                neg_sims = sims[~mask]
                all_neg_sims.extend(neg_sims.cpu().tolist())
        
        import numpy as np
        
        return {
            "mean_pos_similarity": float(np.mean(all_pos_sims)) if all_pos_sims else 0.0,
            "std_pos_similarity": float(np.std(all_pos_sims)) if all_pos_sims else 0.0,
            "mean_neg_similarity": float(np.mean(all_neg_sims)) if all_neg_sims else 0.0,
            "std_neg_similarity": float(np.std(all_neg_sims)) if all_neg_sims else 0.0,
            "similarity_gap": float(np.mean(all_pos_sims) - np.mean(all_neg_sims)) if all_pos_sims and all_neg_sims else 0.0,
        }
    
    def get_model(self) -> BaseMultimodalModel:
        """Retorna o modelo treinado."""
        return self.model
    
    def get_best_checkpoint_path(self) -> Path:
        """Retorna caminho do melhor checkpoint."""
        return self.logger.get_output_dir() / "checkpoints" / "best"

