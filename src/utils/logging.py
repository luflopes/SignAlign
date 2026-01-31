"""Sistema de logging estruturado para experimentos."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


class ExperimentLogger:
    """
    Logger estruturado para experimentos de ML.
    
    Salva métricas em JSON e suporta integração com TensorBoard.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        use_tensorboard: bool = True
    ):
        """
        Inicializa o logger.
        
        Args:
            experiment_name: Nome único do experimento.
            output_dir: Diretório base para salvar logs.
            use_tensorboard: Se True, também loga no TensorBoard.
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.metrics_dir = self.output_dir / "metrics"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        # Criar diretórios
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        # Histórico de métricas
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError:
                print("⚠️ TensorBoard não disponível. Continuando sem ele.")
        
        # Metadata do experimento
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "best_metric": None,
            "best_epoch": None,
        }
        
        print(f"📁 Logs serão salvos em: {self.output_dir}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Salva configuração do experimento."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"📝 Configuração salva em: {config_path}")
    
    def log_train_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        lr: float,
        extra_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Loga métricas de um passo de treinamento."""
        metrics = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        
        self.train_history.append(metrics)
        
        if self.writer:
            global_step = epoch * 1000 + step  # Aproximação
            self.writer.add_scalar("train/loss", loss, global_step)
            self.writer.add_scalar("train/lr", lr, global_step)
            if extra_metrics:
                for key, value in extra_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, global_step)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        lr: float
    ) -> None:
        """Loga métricas de uma época completa."""
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        self.val_history.append(epoch_log)
        
        if self.writer:
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/lr", lr, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"epoch/val_{key}", value, epoch)
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float) -> None:
        """Registra informações sobre o melhor modelo."""
        self.metadata["best_epoch"] = epoch
        self.metadata["best_metric"] = {metric_name: metric_value}
    
    def save_checkpoint(
        self,
        model,
        processor,
        epoch: int,
        metric_value: float,
        is_best: bool = False
    ) -> Path:
        """
        Salva checkpoint do modelo.
        
        Args:
            model: Modelo a ser salvo.
            processor: Processor do modelo.
            epoch: Época atual.
            metric_value: Valor da métrica principal.
            is_best: Se True, salva também como 'best'.
            
        Returns:
            Caminho onde o checkpoint foi salvo.
        """
        checkpoint_name = f"epoch_{epoch:03d}_acc_{metric_value:.4f}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        
        if is_best:
            best_path = self.checkpoints_dir / "best"
            if best_path.exists():
                import shutil
                shutil.rmtree(best_path)
            best_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"🌟 Melhor modelo salvo em: {best_path}")
        
        return checkpoint_path
    
    def save_metrics(self) -> None:
        """Salva histórico de métricas em JSON."""
        train_path = self.metrics_dir / "train_history.json"
        val_path = self.metrics_dir / "val_history.json"
        
        with open(train_path, "w") as f:
            json.dump(self.train_history, f, indent=2)
        
        with open(val_path, "w") as f:
            json.dump(self.val_history, f, indent=2)
    
    def finalize(self) -> None:
        """Finaliza o logging e salva todos os dados."""
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Salvar metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Salvar histórico
        self.save_metrics()
        
        if self.writer:
            self.writer.close()
        
        print(f"✅ Experimento finalizado. Resultados em: {self.output_dir}")
    
    def get_output_dir(self) -> Path:
        """Retorna diretório de saída."""
        return self.output_dir
    
    def get_visualizations_dir(self) -> Path:
        """Retorna diretório de visualizações."""
        return self.visualizations_dir

