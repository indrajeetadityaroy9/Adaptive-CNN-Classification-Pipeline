import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelEvaluator:

    def __init__(self, model, device,
                 class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.results = {}

    @torch.no_grad()
    def evaluate(self, data_loader, criterion=None):
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        logger.info("Starting evaluation...")

        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        metrics = self._calculate_metrics(
            all_targets, all_predictions, all_probabilities
        )

        metrics['loss'] = total_loss / len(data_loader)

        self.results = {
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'metrics': metrics
        }

        return metrics

    def _calculate_metrics(self, targets, predictions,
                          probabilities):
        metrics = {}

        metrics['accuracy'] = np.mean(predictions == targets)
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro')
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro')
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro')

        metrics['precision_per_class'] = precision_score(
            targets, predictions, average=None
        )
        metrics['recall_per_class'] = recall_score(
            targets, predictions, average=None
        )
        metrics['f1_per_class'] = f1_score(
            targets, predictions, average=None
        )

        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)

        for k in [1, 3, 5]:
            if k <= probabilities.shape[1]:
                metrics[f'top_{k}_accuracy'] = self._top_k_accuracy(
                    targets, probabilities, k
                )

        metrics['classification_report'] = classification_report(
            targets, predictions,
            target_names=self.class_names if self.class_names else None,
            output_dict=True
        )

        num_classes = probabilities.shape[1]
        targets_one_hot = np.eye(num_classes)[targets]

        metrics['auc_macro'] = roc_auc_score(
            targets_one_hot, probabilities, average='macro'
        )
        metrics['auc_weighted'] = roc_auc_score(
            targets_one_hot, probabilities, average='weighted'
        )

        return metrics

    def _top_k_accuracy(self, targets, probabilities,
                        k):
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
        return correct / len(targets)

    def plot_confusion_matrix(self, save_path=None,
                             normalize=False):
        if 'confusion_matrix' not in self.results.get('metrics', {}):
            logger.error("No confusion matrix available. Run evaluate() first.")
            return

        cm = self.results['metrics']['confusion_matrix']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else 'auto',
            yticklabels=self.class_names if self.class_names else 'auto'
        )
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_class_distribution(self, save_path=None):
        if not self.results:
            logger.error("No results available. Run evaluate() first.")
            return

        predictions = self.results['predictions']
        targets = self.results['targets']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unique, counts = np.unique(targets, return_counts=True)
        axes[0].bar(unique, counts, alpha=0.7)
        axes[0].set_title('True Label Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        if self.class_names:
            axes[0].set_xticks(unique)
            axes[0].set_xticklabels(self.class_names, rotation=45)

        unique, counts = np.unique(predictions, return_counts=True)
        axes[1].bar(unique, counts, alpha=0.7, color='orange')
        axes[1].set_title('Predicted Label Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        if self.class_names:
            axes[1].set_xticks(unique)
            axes[1].set_xticklabels(self.class_names, rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")

        plt.show()

    def plot_roc_curves(self, save_path=None):
        if not self.results:
            logger.error("No results available. Run evaluate() first.")
            return

        probabilities = self.results['probabilities']
        targets = self.results['targets']
        num_classes = probabilities.shape[1]

        targets_one_hot = np.eye(num_classes)[targets]

        plt.figure(figsize=(10, 8))

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(targets_one_hot[:, i], probabilities[:, i])
            auc = roc_auc_score(targets_one_hot[:, i], probabilities[:, i])

            label = f'{self.class_names[i]} (AUC = {auc:.3f})' if self.class_names else f'Class {i} (AUC = {auc:.3f})'
            plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")

        plt.show()

    def get_misclassified_samples(self, n_samples=10):
        if not self.results:
            logger.error("No results available. Run evaluate() first.")
            return []

        predictions = self.results['predictions']
        targets = self.results['targets']
        probabilities = self.results['probabilities']

        misclassified_mask = predictions != targets
        misclassified_indices = np.where(misclassified_mask)[0]

        samples = []
        for idx in misclassified_indices[:n_samples]:
            sample = {
                'index': int(idx),
                'true_label': int(targets[idx]),
                'predicted_label': int(predictions[idx]),
                'confidence': float(probabilities[idx, predictions[idx]]),
                'true_label_confidence': float(probabilities[idx, targets[idx]])
            }

            if self.class_names:
                sample['true_class'] = self.class_names[targets[idx]]
                sample['predicted_class'] = self.class_names[predictions[idx]]

            samples.append(sample)

        return samples

    def print_summary(self):
        if 'metrics' not in self.results:
            logger.error("No metrics available. Run evaluate() first.")
            return

        metrics = self.results['metrics']

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print("\nOverall Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")

        if 'loss' in metrics:
            print(f"Loss: {metrics['loss']:.4f}")

        print("\nTop-K Accuracy:")
        for k in [1, 3, 5]:
            if f'top_{k}_accuracy' in metrics:
                print(f"Top-{k}: {metrics[f'top_{k}_accuracy']:.4f}")

        if 'auc_macro' in metrics:
            print("\nAUC Scores:")
            print(f"Macro: {metrics['auc_macro']:.4f}")
            print(f"Weighted: {metrics['auc_weighted']:.4f}")

        if self.class_names:
            print("\nPer-Class Metrics:")
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 45)
            for i, class_name in enumerate(self.class_names):
                print(f"{class_name:<15} "
                      f"{metrics['precision_per_class'][i]:<10.4f} "
                      f"{metrics['recall_per_class'][i]:<10.4f} "
                      f"{metrics['f1_per_class'][i]:<10.4f}")

        print("=" * 60)

class TestTimeAugmentation:

    def __init__(self, model, device,
                 n_augmentations=10):
        self.model = model
        self.device = device
        self.n_augmentations = n_augmentations

    def predict(self, image, transforms):
        self.model.eval()
        predictions = []

        with torch.no_grad():

            image = image.to(self.device)
            output = self.model(image)
            predictions.append(torch.softmax(output, dim=1))

            for _ in range(self.n_augmentations - 1):
                aug_image = image.clone()
                for transform in transforms:
                    aug_image = transform(aug_image)

                output = self.model(aug_image)
                predictions.append(torch.softmax(output, dim=1))

        avg_predictions = torch.stack(predictions).mean(dim=0)
        return avg_predictions
