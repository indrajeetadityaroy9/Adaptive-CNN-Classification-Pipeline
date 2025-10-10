import argparse
import os
import random
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.config import (
    ConfigManager,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
    setup_logging
)
from src.models import ModelFactory
from src.datasets import DatasetManager, preprocess_image
from src.training import Trainer
from src.evaluation import ModelEvaluator, TestTimeAugmentation
from src.visualization import (
    FeatureMapVisualizer, GradCAM, TrainingVisualizer
)

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(config_path, resume=None):
    config = ConfigManager.load_config(config_path)

    setup_logging(config.logging)
    logger.info(f"Starting training with config: {config.experiment_name}")

    set_seed(config.seed)

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model = ModelFactory.create_model(config.model.model_type, config.model.__dict__)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    train_loader, val_loader, test_loader = DatasetManager.create_data_loaders(config)

    trainer = Trainer(model, config, device)

    if resume:
        trainer.load_checkpoint(resume)
        logger.info(f"Resumed from checkpoint: {resume}")

    trainer.train(train_loader, val_loader)

    logger.info("Evaluating on test set...")
    evaluator = ModelEvaluator(model, device, dataset_info.get('classes'))
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_summary()

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ConfigManager.save_config(config, output_dir / 'config.yaml')

    if trainer.metrics_history:
        TrainingVisualizer.plot_training_history(
            trainer.metrics_history,
            save_path=output_dir / 'training_history.png'
        )

    evaluator.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True
    )

    logger.info(f"Training completed. Results saved to {output_dir}")

def evaluate_model(config_path, checkpoint_path):

    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model = ModelFactory.create_model(config.model.model_type, config.model.__dict__)
    model = model.to(device)

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Config,
            DataConfig,
            ModelConfig,
            TrainingConfig,
            LoggingConfig
        ])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {checkpoint_path}")

    _, _, test_loader = DatasetManager.create_data_loaders(config)

    evaluator = ModelEvaluator(model, device, dataset_info.get('classes'))
    test_metrics = evaluator.evaluate(test_loader)

    evaluator.print_summary()

    output_dir = Path(config.output_dir) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_confusion_matrix(
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True
    )

    evaluator.plot_roc_curves(save_path=output_dir / 'roc_curves.png')

    evaluator.plot_class_distribution(save_path=output_dir / 'class_distribution.png')

    misclassified = evaluator.get_misclassified_samples(n_samples=20)
    logger.info(f"Top misclassified samples: {misclassified[:5]}")

def test_single_image(config_path, checkpoint_path, image_path, use_tta=False):

    config = ConfigManager.load_config(config_path)
    setup_logging(config.logging)

    device = torch.device(config.device)

    dataset_info = DatasetManager.get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']
    config.model.dataset = config.data.dataset

    model = ModelFactory.create_model(config.model.model_type, config.model.__dict__)
    model = model.to(device)

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Config,
            DataConfig,
            ModelConfig,
            TrainingConfig,
            LoggingConfig
        ])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image = preprocess_image(image_path, config)
    image = image.to(device)

    if use_tta:
        tta = TestTimeAugmentation(model, device, n_augmentations=10)

        transforms = []
        output = tta.predict(image, transforms)
    else:
        with torch.no_grad():
            output = model(image)
            output = torch.softmax(output, dim=1)

    pred_prob, pred_class = output.max(1)
    class_names = dataset_info.get('classes', [str(i) for i in range(dataset_info['num_classes'])])

    print(f"\nPrediction Results:")
    print(f"  Predicted Class: {class_names[pred_class.item()]}")
    print(f"  Confidence: {pred_prob.item():.4f}")

    top5_probs, top5_classes = output.topk(5, dim=1)
    print(f"\nTop-5 Predictions:")
    for i in range(5):
        class_idx = top5_classes[0, i].item()
        prob = top5_probs[0, i].item()
        print(f"  {i+1}. {class_names[class_idx]}: {prob:.4f}")

    output_dir = Path('outputs') / 'inference'
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = FeatureMapVisualizer(model, device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            visualizer.visualize_feature_maps(
                image, name, n_features=32,
                save_path=output_dir / f'feature_maps_{name.replace("/", "_")}.png'
            )
            break

    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name

    if last_conv_name:

        from PIL import Image
        original_img = Image.open(image_path)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        original_img = np.array(original_img)

        gradcam = GradCAM(model, last_conv_name, device)
        gradcam.visualize(
            image, original_img,
            class_idx=pred_class.item(),
            save_path=output_dir / 'gradcam.png'
        )

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive CNN Training System',
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('config', type=str, help='Path to configuration file')
    train_parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('config', type=str, help='Path to configuration file')
    eval_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')

    test_parser = subparsers.add_parser('test', help='Test on single image')
    test_parser.add_argument('config', type=str, help='Path to configuration file')
    test_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    test_parser.add_argument('image', type=str, help='Path to input image')
    test_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    config_parser = subparsers.add_parser('create-config', help='Create default configuration')
    config_parser.add_argument('dataset', choices=['mnist', 'cifar'],
                              help='Dataset to create config for')
    config_parser.add_argument('--output', type=str, default='config.yaml',
                              help='Output path for configuration file')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args.config, args.resume)
    elif args.command == 'evaluate':
        evaluate_model(args.config, args.checkpoint)
    elif args.command == 'test':
        test_single_image(args.config, args.checkpoint, args.image, args.tta)
    elif args.command == 'create-config':
        config = ConfigManager.get_default_config(args.dataset)
        ConfigManager.save_config(config, args.output)
        print(f"Default configuration saved to {args.output}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
