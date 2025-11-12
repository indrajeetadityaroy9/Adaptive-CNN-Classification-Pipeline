import argparse
import os
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
from src.evaluation import ModelEvaluator, TestTimeAugmentation
from src.visualization import (
    FeatureMapVisualizer, GradCAM, TrainingVisualizer
)

logger = logging.getLogger(__name__)

def train_on_modal(config_path, num_gpus=1, experiment_name=None):
    import subprocess

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"\n{'='*60}")
    print("Starting Modal GPU Training")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"GPUs: {num_gpus}")
    print(f"Experiment: {experiment_name or 'auto-generated'}")
    print(f"{'='*60}\n")

    cmd = [
        "modal", "run", "modal_train.py",
        "--config-path", config_path,
        "--num-gpus", str(num_gpus)
    ]

    if experiment_name:
        cmd.extend(["--experiment-name", experiment_name])

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"{'='*60}\n")
        print("To download results:")
        print(f"  python main.py download --list")
        print(f"  python main.py download --experiment <name> --output ./results")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print("Training failed!")
        print(f"{'='*60}\n")
        print(f"Error: {e}")
        return e.returncode
    except FileNotFoundError:
        return 1

def download_from_modal(list_only=False, experiment_name=None, checkpoint_name=None, output_dir='./downloads'):
    import subprocess

    VOLUME_NAME = "cnn-training-vol"

    try:
        if list_only:
            print(f"\n{'='*60}")
            print("Available Experiments and Checkpoints")
            print(f"{'='*60}\n")

            print("Experiments (outputs):")
            print("-" * 60)
            result = subprocess.run(
                ["modal", "volume", "ls", VOLUME_NAME, "/vol/outputs"],
                capture_output=True, text=True, check=True
            )
            print(result.stdout if result.stdout else "No experiments found")

            print("\nCheckpoints:")
            print("-" * 60)
            result = subprocess.run(
                ["modal", "volume", "ls", VOLUME_NAME, "/vol/checkpoints"],
                capture_output=True, text=True, check=True
            )
            print(result.stdout if result.stdout else "No checkpoints found")
            print(f"{'='*60}\n")
            return 0

        if experiment_name:
            remote_path = f"/vol/outputs/{experiment_name}"
            local_path = Path(output_dir) / experiment_name
            local_path.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Downloading experiment: {experiment_name}")
            print(f"{'='*60}")
            print(f"From: {VOLUME_NAME}:{remote_path}")
            print(f"To: {local_path}")
            print(f"{'='*60}\n")

            result = subprocess.run(
                ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_path)],
                check=True
            )

            print(f"\n{'='*60}")
            print("Download completed!")
            print(f"{'='*60}")
            print(f"Files saved to: {local_path}")
            print(f"\nContents:")
            for item in local_path.rglob('*'):
                if item.is_file():
                    print(f"  - {item.relative_to(local_path)}")
            print(f"{'='*60}\n")
            return result.returncode

        if checkpoint_name:
            remote_path = f"/vol/checkpoints/{checkpoint_name}"
            local_path = Path(output_dir) / "checkpoints"
            local_path.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Downloading checkpoint: {checkpoint_name}")
            print(f"{'='*60}")
            print(f"From: {VOLUME_NAME}:{remote_path}")
            print(f"To: {local_path / checkpoint_name}")
            print(f"{'='*60}\n")

            result = subprocess.run(
                ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_path / checkpoint_name)],
                check=True
            )

            print(f"\n{'='*60}")
            print("Download completed!")
            print(f"{'='*60}")
            print(f"Checkpoint saved to: {local_path / checkpoint_name}")
            print(f"\nTo use for evaluation:")
            print(f"  python main.py evaluate <config> {local_path / checkpoint_name}")
            print(f"{'='*60}\n")
            return result.returncode
        return 1

    except subprocess.CalledProcessError as e:
        print(f"\nError downloading from Modal: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        return e.returncode

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

    train_parser = subparsers.add_parser('train', help='Train a model on Modal GPU infrastructure')
    train_parser.add_argument('config', type=str, help='Path to configuration file')
    train_parser.add_argument('--num-gpus', type=int, default=1,
                             help='Number of GPUs to use (1-8, default: 1)')
    train_parser.add_argument('--experiment-name', type=str,
                             help='Optional experiment name')

    download_parser = subparsers.add_parser('download', help='Download model weights and artifacts from Modal')
    download_parser.add_argument('--list', action='store_true', dest='list_only',
                                help='List available experiments and checkpoints')
    download_parser.add_argument('--experiment', type=str, dest='experiment_name',
                                help='Download entire experiment output directory')
    download_parser.add_argument('--checkpoint', type=str, dest='checkpoint_name',
                                help='Download specific checkpoint file')
    download_parser.add_argument('--output', type=str, default='./downloads',
                                help='Local directory to save downloads (default: ./downloads)')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model locally')
    eval_parser.add_argument('config', type=str, help='Path to configuration file')
    eval_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')

    test_parser = subparsers.add_parser('test', help='Test on single image with visualization')
    test_parser.add_argument('config', type=str, help='Path to configuration file')
    test_parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    test_parser.add_argument('image', type=str, help='Path to input image')
    test_parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    args = parser.parse_args()

    if args.command == 'train':
        train_on_modal(args.config, args.num_gpus, args.experiment_name)
    elif args.command == 'download':
        download_from_modal(args.list_only, args.experiment_name, args.checkpoint_name, args.output)
    elif args.command == 'evaluate':
        evaluate_model(args.config, args.checkpoint)
    elif args.command == 'test':
        test_single_image(args.config, args.checkpoint, args.image, args.tta)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
