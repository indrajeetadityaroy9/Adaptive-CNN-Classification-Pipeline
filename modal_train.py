import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import modal
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if '/root' not in sys.path and Path('/root').exists():
    sys.path.insert(0, '/root')

from src.config import ConfigManager, setup_logging
from src.datasets import DatasetManager
from src.evaluation import ModelEvaluator
from src.models import ModelFactory
from src.training import DDPTrainer
from src.visualization import TrainingVisualizer

volume = modal.Volume.from_name("cnn-training-vol", create_if_missing=True)

VOLUME_PATH = Path("/vol")
DATA_PATH = VOLUME_PATH / "data"
CHECKPOINTS_PATH = VOLUME_PATH / "checkpoints"
OUTPUTS_PATH = VOLUME_PATH / "outputs"

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "opencv-python>=4.8.0",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App("adaptive-cnn-training", image=training_image)

def train_worker(
    rank,
    world_size,
    config_path,
    experiment_name: Optional[str]
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    is_main_process = (rank == 0)

    if is_main_process:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
        OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    config_full_path = Path("/root") / config_path
    if not config_full_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ConfigManager.load_config(str(config_full_path))

    config.data.data_path = str(DATA_PATH)
    config.output_dir = str(OUTPUTS_PATH)
    config.logging.log_dir = str(OUTPUTS_PATH / "logs")

    if experiment_name:
        config.experiment_name = experiment_name
    config.experiment_name = f"{config.experiment_name}_ddp_{world_size}gpu"

    if is_main_process:
        setup_logging(config.logging)
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("PyTorch Distributed Data Parallel (DDP) Training")
        logger.info("=" * 60)
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Dataset: {config.data.dataset}")
        logger.info(f"World Size: {world_size}")
        logger.info(f"GPUs: {world_size} x A100 (40GB each)")
        logger.info("Backend: NCCL")
        logger.info("=" * 60)
    else:
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    set_seed(config.seed + rank)

    if is_main_process:
        logger.info(f"Process {rank}: Using device cuda:{rank}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(rank)}")
        logger.info(f"Total GPUs Available: {torch.cuda.device_count()}")

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model = ModelFactory.create_model(config.model.model_type, config.model.__dict__)
    model = model.to(device)

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    if is_main_process:
        total_params = sum(p.numel() for p in model.module.parameters())
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        logger.info("Model wrapped with DistributedDataParallel")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Effective batch size: {config.data.batch_size * world_size}")

    if is_main_process:
        logger.info("Creating distributed data loaders...")

    train_dataset = DatasetManager.get_dataset(config, is_train=True)
    val_dataset = DatasetManager.get_dataset(config, is_train=False)
    test_dataset = DatasetManager.get_dataset(config, is_train=False)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else 2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    if is_main_process:
        logger.info(f"Train samples: {len(train_dataset)} ({len(train_dataset)//world_size} per GPU)")
        logger.info(f"Val samples: {len(val_dataset)} ({len(val_dataset)//world_size} per GPU)")
        logger.info(f"Test samples: {len(test_dataset)}")

    ddp_trainer = DDPTrainer(model, config, device, rank, world_size)

    checkpoint_path = CHECKPOINTS_PATH / f"best_model_{config.data.dataset}_ddp.pth"

    if is_main_process:
        if checkpoint_path.exists():
            logger.info(f"Found checkpoint: {checkpoint_path}")
            try:
                ddp_trainer.load_checkpoint(checkpoint_path)
                logger.info(f"Resumed from epoch {ddp_trainer.current_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        else:
            logger.info("No checkpoint found. Starting fresh training.")

    dist.barrier()

    if is_main_process:
        logger.info("=" * 60)
        logger.info("STARTING DDP TRAINING")
        logger.info("=" * 60)

    metrics_history = ddp_trainer.train_ddp(train_loader, train_sampler, val_loader)

    if is_main_process:
        logger.info("=" * 60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 60)

        evaluator = ModelEvaluator(ddp_trainer.model, device, dataset_info.get('classes'))
        test_metrics = evaluator.evaluate(test_loader)
        evaluator.print_summary()
        output_dir = OUTPUTS_PATH / config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        ConfigManager.save_config(config, output_dir / 'config.yaml')

        if metrics_history:
            TrainingVisualizer.plot_training_history(
                metrics_history,
                save_path=output_dir / 'training_history.png'
            )

        evaluator.plot_confusion_matrix(
            save_path=output_dir / 'confusion_matrix.png',
            normalize=True
        )

        metrics_file = output_dir / 'test_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FINAL TEST METRICS (DDP Training)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Configuration: {world_size} GPUs with DDP\n")
            f.write(f"Effective Batch Size: {config.data.batch_size * world_size}\n\n")
            f.write(f"Accuracy:          {test_metrics['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {test_metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (macro):    {test_metrics['recall_macro']:.4f}\n")
            f.write(f"F1 Score (macro):  {test_metrics['f1_macro']:.4f}\n")
            f.write(f"Loss:              {test_metrics.get('loss', 'N/A')}\n")

            if 'auc_macro' in test_metrics:
                f.write(f"\nAUC Macro: {test_metrics['auc_macro']:.4f}\n")
                f.write(f"AUC Weighted: {test_metrics['auc_weighted']:.4f}\n")

        volume.commit()

        logger.info("=" * 60)
        logger.info("DDP TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Outputs saved to: {output_dir}")
        logger.info("=" * 60)

        result = {
            'experiment_name': config.experiment_name,
            'test_accuracy': test_metrics['accuracy'],
            'output_dir': str(output_dir),
            'num_gpus': world_size
        }
    else:
        result = None

    dist.destroy_process_group()
    return result

def train_orchestrator(num_gpus, config_path, experiment_name: Optional[str] = None):
    if num_gpus == 1:
        return train_worker(0, 1, config_path, experiment_name)
    else:
        mp.spawn(train_worker, args=(num_gpus, config_path, experiment_name), nprocs=num_gpus, join=True)
        return {"status": "completed", "num_gpus": num_gpus}


@app.function(
    gpu="a100:8",
    timeout=86400,
    volumes={str(VOLUME_PATH): volume},
    retries=modal.Retries(max_retries=10, initial_delay=0.0, backoff_coefficient=1.0),
)
def train_with_gpus(num_gpus, config_path, experiment_name: Optional[str] = None):
    return train_orchestrator(num_gpus, config_path, experiment_name)


@app.local_entrypoint()
def main(
    config_path,
    experiment_name: Optional[str] = None,
    num_gpus=1
):
    if num_gpus < 1 or num_gpus > 8:
        raise ValueError("num_gpus must be between 1 and 8")

    print(f"\n{'='*60}")
    print("PyTorch Distributed Data Parallel (DDP) Training")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Experiment: {experiment_name or 'auto-generated'}")
    print(f"GPUs: {num_gpus} x A100 (using {num_gpus} of 8 allocated)")
    print(f"Mode: {'DDP Multi-Process' if num_gpus > 1 else 'Single GPU'}")
    print(f"Effective Batch Size: Configured batch_size × {num_gpus}")
    print(f"{'='*60}\n")

    result = train_with_gpus.remote(num_gpus, config_path, experiment_name)

    if result and isinstance(result, dict):
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        if 'test_accuracy' in result:
            print(f"Experiment: {result['experiment_name']}")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"GPUs Used: {result['num_gpus']}")
            print(f"Output Directory: {result['output_dir']}")
            print("\nTo download outputs:")
            print(f"modal volume get cnn-training-vol outputs/{Path(result['output_dir']).name} ./ddp-results")
        else:
            print(f"DDP Training completed with {result.get('num_gpus', num_gpus)} GPUs")
            print("\nTo download outputs:")
            print("modal volume ls cnn-training-vol outputs")
        print(f"{'='*60}\n")
