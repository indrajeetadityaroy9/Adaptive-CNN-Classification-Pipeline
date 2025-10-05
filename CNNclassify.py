import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dataset):
        super(CNNClassifier, self).__init__()

        if dataset == 'mnist':
            self.C1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
            self.R1 = nn.BatchNorm2d(32)
            self.C2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.R2 = nn.BatchNorm2d(64)
            self.P = nn.AdaptiveAvgPool2d((4, 4))
            self.F1 = nn.Linear(64 * 4 * 4, 128)
            self.O = nn.Linear(128, num_classes)
        elif dataset == 'cifar':
            self.C1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
            self.R1 = nn.BatchNorm2d(32)
            self.C2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.R2 = nn.BatchNorm2d(64)
            self.C3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.R3 = nn.BatchNorm2d(128)
            self.C4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.R4 = nn.BatchNorm2d(256)
            self.P = nn.AdaptiveAvgPool2d((4, 4))
            self.F1 = nn.Linear(256 * 4 * 4, 512)
            self.F2 = nn.Linear(512, 128)
            self.O = nn.Linear(128, num_classes)
        self.D = nn.Dropout(0.5)

    def forward(self, x, dataset):
        if dataset == 'mnist':
            x = F.relu(self.R1(self.C1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.R2(self.C2(x)))
            x = F.max_pool2d(x, 2)
            x = self.P(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.F1(x))
            x = self.D(x)
            x = self.O(x)
        elif dataset == 'cifar':
            x = F.relu(self.R1(self.C1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.R2(self.C2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.R3(self.C3(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.R4(self.C4(x)))
            x = self.P(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.F1(x))
            x = self.D(x)
            x = F.relu(self.F2(x))
            x = self.D(x)
            x = self.O(x)
        return x


def train_model(model, train_data_loader, test_data_loader, loss_func, optimizer, lr_scheduler, device, num_epochs, dataset):
    model_directory = "model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename = f"{dataset}_trained_model.pth"
    best_model_filepath = os.path.join(model_directory, model_filename)

    best_test_accuracy = 0.0

    print(f"{'Epoch':>6}{'Train Loss':>15}{'Train Acc %':>20}{'Test Loss':>20}{'Test Acc %':>20}")
    for epoch in range(num_epochs):
        model.train()
        running_correct_predictions = 0
        running_total_samples = 0
        accumulated_train_loss = 0.0

        with tqdm(train_data_loader, unit="batch", leave=False) as train_progress:
            train_progress.set_description(f"Epoch [{epoch + 1}/{num_epochs}] - Training")
            for batch_idx, (input_images, labels) in enumerate(train_progress):
                input_images, labels = input_images.to(device), labels.to(device)
                running_total_samples += labels.size(0)

                predictions = model(input_images, dataset)
                loss = loss_func(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                accumulated_train_loss += loss.item()
                running_correct_predictions += (torch.max(predictions, 1)[1] == labels).sum().item()

                train_progress.set_postfix({
                    'Loss': accumulated_train_loss / (batch_idx + 1),
                    'Accuracy': 100. * running_correct_predictions / running_total_samples
                })

        average_train_loss = accumulated_train_loss / len(train_data_loader)
        train_accuracy = 100. * running_correct_predictions / running_total_samples

        model.eval()
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0

        with tqdm(test_data_loader, unit="batch", leave=False) as test_progress:
            test_progress.set_description(f"Epoch [{epoch + 1}/{num_epochs}] - Testing")
            with torch.no_grad():
                for batch_idx, (input_images, labels) in enumerate(test_progress):
                    input_images, labels = input_images.to(device), labels.to(device)

                    predictions = model(input_images, dataset)
                    loss = loss_func(predictions, labels)
                    test_loss += loss.item()
                    _, predicted_labels = torch.max(predictions, 1)
                    total_test_samples += labels.size(0)
                    correct_test_predictions += (predicted_labels == labels).sum().item()
                    test_progress.set_postfix({
                        'Loss': test_loss / (batch_idx + 1),
                        'Accuracy': 100. * correct_test_predictions / total_test_samples
                    })

        average_test_loss = test_loss / len(test_data_loader)
        test_accuracy = 100. * correct_test_predictions / total_test_samples

        print(f"{epoch:>2}/{num_epochs:<1}{average_train_loss:>15f}{train_accuracy:>20f}{average_test_loss:>20f}{test_accuracy:>20f}") if \
            (num_epochs == 10 or (num_epochs == 50 and (epoch % 10 == 0 or epoch == num_epochs - 1))) else None

        lr_scheduler.step(average_test_loss)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_filepath)

    print(f"Model trained on the {dataset} dataset with best test accuracy: {best_test_accuracy:.2f}% saved in file: {best_model_filepath}.")


def get_dataset_info(dataset):
    if dataset == 'cifar':
        cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        labels = cifar_dataset.classes
        input_channels = cifar_dataset[0][0].shape[0]
        num_classes = len(cifar_dataset.classes)
        means = (cifar_dataset.data / 255.0).mean(axis=(0, 1, 2))
        stds = (cifar_dataset.data / 255.0).std(axis=(0, 1, 2))
        return input_channels, num_classes, means, stds, labels

    elif dataset == 'mnist':
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        labels = mnist_dataset.classes
        input_channels = 1
        num_classes = len(mnist_dataset.classes)
        means = (mnist_dataset.data / 255.0).mean()
        stds = (mnist_dataset.data / 255.0).std()
        return input_channels, num_classes, means, stds, labels


def load_dataset(dataset, batch_size, means, stds):
    if dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(means,), std=(stds,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(means,), std=(stds,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    return train_loader, test_loader


def preprocess_image(input_image, dataset, means, stds):
    if dataset == 'mnist':
        image = Image.open(input_image).convert("L")
        image_np = np.array(image)
        mean_brightness = image_np.mean()
        if mean_brightness > 127:
            image = ImageOps.invert(image)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((means,), (stds,))
        ])
    elif dataset == 'cifar':
        image = Image.open(input_image)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

    image = transform(image).unsqueeze(0)
    return image


def classify_image(model, input_image, device, dataset, mean, std, dataset_labels):
    image = preprocess_image(input_image, dataset, mean, std).to(device)

    model.eval()
    with torch.no_grad():
        conv1_output = model.C1(image)
        output = model(image, dataset)
        _, predicted = torch.max(output, 1)
        predicted_class_idx = predicted.item()
    visualize_first_conv_layer(conv1_output, dataset)
    return dataset_labels[predicted_class_idx]


def visualize_first_conv_layer(conv1_output, dataset):
    conv1_output = conv1_output.cpu()
    num_filters = conv1_output.shape[1]
    fig, axes = plt.subplots(4, 8, figsize=(14, 8), dpi=300)

    for i in range(num_filters):
        ax = axes[i // 8, i % 8]
        feature_map = conv1_output[0, i].numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        ax.imshow(feature_map, cmap='gray', interpolation='bicubic')
        ax.axis('off')

    output_filename = f"CONV_rslt_{dataset}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()


def load_saved_model(model_class, num_classes, in_channels, model_directory, dataset, device):
    model_filename = f"{dataset}_trained_model.pth"
    model_path = os.path.join(model_directory, model_filename)

    if not os.path.exists(model_path):
        return None

    model = model_class(in_channels, num_classes, dataset)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, data_loader, device, dataset_name, num_classes, k_values=[1, 5]):
    model.eval()
    all_labels = []
    all_preds = []
    top_k_correct = {k: 0 for k in k_values}

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, dataset_name)

            probs = torch.softmax(outputs, dim=1)
            top_k_preds = {k: probs.topk(k, dim=1).indices for k in k_values}

            for k in k_values:
                top_k_correct[k] += (
                    torch.eq(labels.view(-1, 1), top_k_preds[k]).any(dim=1).sum().item()
                )

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print("\n--- Classification Metrics ---")
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print("\n--- Top-K Accuracy ---")
    total_samples = len(all_labels)
    for k in k_values:
        top_k_acc = top_k_correct[k] / total_samples
        print(f"Top-{k} Accuracy: {top_k_acc:.4f}")


def main():
    epilog = """Usage examples:
      python3 CNNclassify.py train --mnist       Train the model using the MNIST dataset
      python3 CNNclassify.py train --cifar       Train the model using the CIFAR-10 dataset
      python3 CNNclassify.py test car.png        Test the model using 'car.png'
      python3 CNNclassify.py evaluate --mnist    Evaluate the model trained on the MNIST dataset
      python3 CNNclassify.py evaluate --cifar    Evaluate the model trained on the CIFAR-10 dataset
    """

    parser = argparse.ArgumentParser(
        usage='python3 CNNclassify.py [-h] {train,test,evaluate} ...',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        epilog=epilog
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute: train, test, or evaluate")

    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--mnist', action='store_true', help="Train the CNN using the MNIST dataset.")
    train_parser.add_argument('--cifar', action='store_true', help="Train the CNN using the CIFAR-10 dataset.")

    test_parser = subparsers.add_parser('test', help="Test the model")
    test_parser.add_argument('image_file', nargs='*', help="Image file(s) for testing (e.g., car.png)")

    evaluate_parser = subparsers.add_parser('evaluate', help="Evaluate the model")
    evaluate_parser.add_argument('--mnist', action='store_true', help="Evaluate the CNN trained on the MNIST dataset.")
    evaluate_parser.add_argument('--cifar', action='store_true', help="Evaluate the CNN trained on the CIFAR-10 dataset.")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'train':
        if not args.mnist and not args.cifar:
            print("Error: 'train' command requires either --mnist or --cifar argument.", file=sys.stderr)
            sys.exit(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = 'mnist' if args.mnist else 'cifar'

        batch_size = 64
        input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
        train_loader, test_loader = load_dataset(dataset, batch_size, means, stds)
        model = CNNClassifier(input_channels, num_classes, dataset).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)
        num_epochs = 10 if dataset == 'mnist' else 50
        train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs, dataset)

    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            sys.exit(1)

        model_dir = "model"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for img_file in args.image_file:
            datasets = ['cifar', 'mnist']
            for dataset in datasets:
                input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
                model = load_saved_model(CNNClassifier, num_classes, input_channels, model_dir, dataset, device)
                if model is not None:
                    predicted_label = classify_image(model, img_file, device, dataset, means, stds, dataset_labels)
                    print(f"Prediction result by model trained on {dataset.upper()} dataset: {predicted_label}")
                else:
                    print(f"Error: Could not load model for {dataset} dataset.")

    elif args.command == 'evaluate':
        if not args.mnist and not args.cifar:
            print("Error: 'evaluate' command requires either --mnist or --cifar argument.", file=sys.stderr)
            sys.exit(1)

        model_dir = "model"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = 'mnist' if args.mnist else 'cifar'

        input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
        train_loader, test_loader = load_dataset(dataset, batch_size=64, means=means, stds=stds)

        model = load_saved_model(CNNClassifier, num_classes, input_channels, model_dir, dataset, device)
        if model is not None:
            evaluate_model(model, test_loader, device, dataset, num_classes, k_values=[1, 5])
        else:
            print(f"Error: Could not load model for {dataset} dataset.")


if __name__ == '__main__':
    main()