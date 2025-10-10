import os
import json
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataConfig:
    def __init__(self, dataset=None, batch_size=64, num_workers=4, pin_memory=True,
                 persistent_workers=True, prefetch_factor=2, augmentation=None,
                 normalization=None, data_path="./data"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.augmentation = augmentation if augmentation is not None else {}
        self.normalization = normalization if normalization is not None else {}
        self.data_path = data_path

class ModelConfig:
    def __init__(self, model_type="adaptive_cnn", in_channels=1, num_classes=10,
                 dropout=0.5, use_se=True, architecture=None, classifier_layers=None):
        self.model_type = model_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_se = use_se
        self.architecture = architecture if architecture is not None else []
        self.classifier_layers = classifier_layers if classifier_layers is not None else []

class TrainingConfig:
    def __init__(self, num_epochs=10, learning_rate=0.001, weight_decay=0.0005,
                 optimizer="adamw", scheduler="cosine", warmup_epochs=5,
                 gradient_clip_val=1.0, gradient_accumulation_steps=1,
                 use_amp=True, amp_backend="native", early_stopping=True,
                 early_stopping_patience=10, early_stopping_min_delta=0.001,
                 lr_scheduler_params=None, label_smoothing=0.1, use_swa=True,
                 swa_start_epoch=0.75, swa_lr=0.0005):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.amp_backend = amp_backend
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.lr_scheduler_params = lr_scheduler_params if lr_scheduler_params is not None else {}
        self.label_smoothing = label_smoothing
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr

class LoggingConfig:
    def __init__(self, log_level="INFO", log_dir="./logs", wandb=False,
                 wandb_project="adaptive-cnn", wandb_entity=None,
                 save_frequency=5, track_grad_norm=True):
        self.log_level = log_level
        self.log_dir = log_dir
        self.wandb = wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.save_frequency = save_frequency
        self.track_grad_norm = track_grad_norm

class Config:
    def __init__(self, data=None, model=None, training=None, logging=None,
                 experiment_name="default", seed=42, device="cuda", output_dir="./outputs"):
        self.data = data
        self.model = model
        self.training = training
        self.logging = logging
        self.experiment_name = experiment_name
        self.seed = seed
        self.device = device
        self.output_dir = output_dir

class ConfigManager:

    @staticmethod
    def load_config(config_path):
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                raw_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        config = Config(
            data=DataConfig(**raw_config.get('data', {})),
            model=ModelConfig(**raw_config.get('model', {})),
            training=TrainingConfig(**raw_config.get('training', {})),
            logging=LoggingConfig(**raw_config.get('logging', {})),
            **{k: v for k, v in raw_config.items()
               if k not in ['data', 'model', 'training', 'logging']}
        )

        ConfigManager.validate_config(config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    @staticmethod
    def validate_config(config):

        valid_datasets = ['mnist', 'cifar', 'custom']
        if config.data.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {config.data.dataset}")

        if config.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if config.model.dropout < 0 or config.model.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")

        if config.training.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        if config.training.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {config.training.optimizer}")

        valid_schedulers = ['step', 'cosine', 'plateau', 'exponential', 'cyclic']
        if config.training.scheduler.lower() not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {config.training.scheduler}")

    @staticmethod
    def save_config(config, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'logging': config.logging.__dict__,
            'experiment_name': config.experiment_name,
            'seed': config.seed,
            'device': config.device,
            'output_dir': config.output_dir
        }

        if save_path.suffix in ['.yml', '.yaml']:
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

        logger.info(f"Saved configuration to {save_path}")

    @staticmethod
    def get_default_config(dataset):
        if dataset == 'mnist':
            return Config(
                data=DataConfig(
                    dataset='mnist',
                    batch_size=64,
                    augmentation={},
                    normalization={'mean': [0.1307], 'std': [0.3081]}
                ),
                model=ModelConfig(
                    in_channels=1,
                    num_classes=10,
                    dropout=0.5
                ),
                training=TrainingConfig(
                    num_epochs=10,
                    learning_rate=0.001,
                    weight_decay=0.0005,
                    lr_scheduler_params={'step_size': 5, 'gamma': 0.1}
                ),
                logging=LoggingConfig()
            )
        elif dataset == 'cifar':
            return Config(
                data=DataConfig(
                    dataset='cifar',
                    batch_size=128,
                    augmentation={
                        'random_crop': True,
                        'random_flip': True,
                        'color_jitter': True,
                        'cutout': True
                    },
                    normalization={
                        'mean': [0.4914, 0.4822, 0.4465],
                        'std': [0.2470, 0.2435, 0.2616]
                    }
                ),
                model=ModelConfig(
                    in_channels=3,
                    num_classes=10,
                    dropout=0.5
                ),
                training=TrainingConfig(
                    num_epochs=200,
                    learning_rate=0.1,
                    weight_decay=0.0005,
                    optimizer='sgd',
                    scheduler='cosine',
                    warmup_epochs=10,
                    lr_scheduler_params={'T_max': 200, 'eta_min': 0.0001}
                ),
                logging=LoggingConfig()
            )
        else:
            raise ValueError(f"No default config for dataset: {dataset}")

def setup_logging(config):
    log_level = getattr(logging, config.log_level.upper())

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.log_dir) / 'training.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("Logging configured successfully")
