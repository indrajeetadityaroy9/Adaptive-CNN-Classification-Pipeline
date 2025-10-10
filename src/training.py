import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict

from src.config import Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig
logger = logging.getLogger(__name__)

class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)
        loss = -log_pred.sum(dim=-1)
        nll = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = loss / n_classes
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()

class Trainer:

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        if config.training.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(config.training.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()

        self.use_amp = config.training.use_amp and device.type == 'cuda'
        self.autocast_kwargs = {}
        if self.use_amp:
            self.scaler = GradScaler(device_type=device.type)
            self.autocast_kwargs = {'device_type': device.type}
            logger.info("Using Automatic Mixed Precision (AMP)")

        self.use_swa = config.training.use_swa
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=config.training.swa_lr
            )
            self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)

        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta
            )
        else:
            self.early_stopping = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        self.grad_accum_steps = config.training.gradient_accumulation_steps

        self.metrics_history = defaultdict(list)

    def _create_optimizer(self):
        opt_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay

        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd,
                           momentum=0.9, nesterov=True)
        elif opt_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _create_scheduler(self):
        scheduler_name = self.config.training.scheduler.lower()
        params = self.config.training.lr_scheduler_params

        if scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get('step_size', 10),
                gamma=params.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get('T_max', self.config.training.num_epochs),
                eta_min=params.get('eta_min', 0.0001)
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=params.get('factor', 0.1),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-7)
            )
        elif scheduler_name == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=params.get('gamma', 0.95)
            )
        elif scheduler_name == 'cyclic':
            return optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=params.get('base_lr', 0.0001),
                max_lr=params.get('max_lr', self.config.training.learning_rate),
                step_size_up=params.get('step_size_up', 2000),
                mode=params.get('mode', 'triangular2')
            )
        else:
            return None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with autocast(**self.autocast_kwargs):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:

                    if self.config.training.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_val
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:

                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip_val
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_count += 1

            current_loss = total_loss / batch_count
            current_acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

            self.global_step += 1

        metrics = {
            'train_loss': total_loss / batch_count,
            'train_acc': 100. * correct / total
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }

        return metrics

    def train(self, train_loader, val_loader, num_epochs=None):
        num_epochs = num_epochs or self.config.training.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            train_metrics = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)

            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            elif self.scheduler is not None:

                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.2f}%, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )

            for key, value in {**train_metrics, **val_metrics}.items():
                self.metrics_history[key].append(value)

            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)

            if (epoch + 1) % self.config.logging.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

            if self.early_stopping:
                if self.early_stopping(val_metrics['val_loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        if self.use_swa:
            logger.info("Updating batch normalization statistics for SWA model")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        logger.info("Training completed")

    def save_checkpoint(self, filename, epoch, metrics):
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'metrics_history': dict(self.metrics_history)
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.use_swa:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()

        save_path = checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                Config,
                DataConfig,
                ModelConfig,
                TrainingConfig,
                LoggingConfig
            ])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.use_swa and 'swa_model_state_dict' in checkpoint:
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
