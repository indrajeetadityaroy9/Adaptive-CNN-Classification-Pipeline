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


# -------------------- Model Architecture --------------------
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dataset):
        super(CNNClassifier, self).__init__()

        if dataset == 'mnist':
            # C (Conv Layer): First convolutional layer
            self.C1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
            # R (ReLU layer): Activation after the first convolution
            self.R1 = nn.BatchNorm2d(32)
            # C (Conv Layer): Second convolutional layer
            self.C2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            # R (ReLU layer): Activation after the second convolution
            self.R2 = nn.BatchNorm2d(64)
            # P (Pooling Layer): Adaptive pooling layer to reduce the size of the feature map
            self.P = nn.AdaptiveAvgPool2d((4, 4))
            # F (Fully Connected Layer): First fully connected layer for classification
            self.F1 = nn.Linear(64 * 4 * 4, 128)
            # O (Output Layer): Final output layer for classification
            self.O = nn.Linear(128, num_classes)
        elif dataset == 'cifar':
            # C (Conv Layer): First convolutional layer
            self.C1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
            # R (ReLU layer): Activation after the first convolution
            self.R1 = nn.BatchNorm2d(32)
            # C (Conv Layer): Second convolutional layer
            self.C2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            # R (ReLU layer): Activation after the second convolution
            self.R2 = nn.BatchNorm2d(64)
            # C (Conv Layer): Third convolutional layer
            self.C3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            # R (ReLU layer): Activation after the third convolution
            self.R3 = nn.BatchNorm2d(128)
            # C (Conv Layer): Fourth convolutional layer
            self.C4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            # R (ReLU layer): Activation after the fourth convolution
            self.R4 = nn.BatchNorm2d(256)
            # P (Pooling Layer): Adaptive pooling layer to reduce the size of the feature map
            self.P = nn.AdaptiveAvgPool2d((4, 4))
            # F (Fully Connected Layer): First fully connected layer for classification
            self.F1 = nn.Linear(256 * 4 * 4, 512)
            # F (Fully Connected Layer): Second fully connected layer for classification
            self.F2 = nn.Linear(512, 128)
            # O (Output Layer): Final output layer for classification
            self.O = nn.Linear(128, num_classes)
        # Dropout (to reduce overfitting)
        self.D = nn.Dropout(0.5)

    def forward(self, x, dataset):
        if dataset == 'mnist':
            # C -> R -> P sequence
            x = F.relu(self.R1(self.C1(x)))  # C1 -> R1
            x = F.max_pool2d(x, 2)  # P
            x = F.relu(self.R2(self.C2(x)))  # C2 -> R2
            x = F.max_pool2d(x, 2)  # P
            x = self.P(x)  # P (feature reduction)
            x = x.view(x.size(0), -1)
            # Fully connected layers
            x = F.relu(self.F1(x))  # F1
            x = self.D(x)  # Dropout (D)
            x = self.O(x)  # O (output)
        elif dataset == 'cifar':
            # C -> R -> P sequence
            x = F.relu(self.R1(self.C1(x)))  # C1 -> R1
            x = F.max_pool2d(x, 2)  # P
            x = F.relu(self.R2(self.C2(x)))  # C2 -> R2
            x = F.max_pool2d(x, 2)  # P
            x = F.relu(self.R3(self.C3(x)))  # C3 -> R3
            x = F.max_pool2d(x, 2)  # P
            x = F.relu(self.R4(self.C4(x)))  # C4 -> R4
            x = self.P(x)  # P (feature reduction)
            x = x.view(x.size(0), -1)
            # Fully connected layers
            x = F.relu(self.F1(x))  # F1
            x = self.D(x)  # Dropout (D)
            x = F.relu(self.F2(x))  # F2
            x = self.D(x)  # Dropout (D)
            x = self.O(x)  # O (output)
        return x


# -------------------- Model Training --------------------
def train_model(model, train_data_loader, test_data_loader, loss_func, optimizer, lr_scheduler, device, num_epochs, dataset):
    """
    Trains model on a specified dataset

    Parameters:
        - model: Model to be trained
        - train_data_loader (DataLoader): DataLoader for the training dataset
        - test_data_loader (DataLoader): DataLoader for the testing/validation dataset
        - loss_func: Loss function used to compute model error
        - optimizer: Optimizer used for gradient descent
        - lr_scheduler: Learning rate scheduler to adjust learning rate during training
        - device: Device to run the model on (CPU/GPU)
        - num_epochs (int): Number of epochs to train the model
        - dataset (str): Dataset being used for training

    Returns:
        - None: Saves the best model during training based on test accuracy
    """

    # Directory where the model will be saved
    model_directory = "model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Path to save the best model
    model_filename = f"{dataset}_trained_model.pth"
    best_model_filepath = os.path.join(model_directory, model_filename)

    # Track the best test accuracy
    best_test_accuracy = 0.0

    print(f"{'Epoch':>6}{'Train Loss':>15}{'Train Acc %':>20}{'Test Loss':>20}{'Test Acc %':>20}")
    # Training loop for each epoch
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_correct_predictions = 0  # Track correct predictions for accuracy
        running_total_samples = 0  # Total samples processed
        accumulated_train_loss = 0.0  # Accumulate loss for averaging

        # Iterate through training batches
        with tqdm(train_data_loader, unit="batch", leave=False) as train_progress:
            train_progress.set_description(f"Epoch [{epoch + 1}/{num_epochs}] - Training")
            for batch_idx, (input_images, labels) in enumerate(train_progress):
                input_images, labels = input_images.to(device), labels.to(device)
                running_total_samples += labels.size(0)  # Update sample count

                # Forward pass: Compute model predictions and calculate loss
                predictions = model(input_images, dataset)
                loss = loss_func(predictions, labels)
                # Backward pass: Clear gradients, compute new gradients, and update model weights
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()  # Update model weights
                # Accumulate training loss
                accumulated_train_loss += loss.item()
                # Count correct predictions for accuracy
                running_correct_predictions += (torch.max(predictions, 1)[1] == labels).sum().item()

                # Update progress bar with running metrics
                train_progress.set_postfix({
                    'Loss': accumulated_train_loss / (batch_idx + 1),
                    'Accuracy': 100. * running_correct_predictions / running_total_samples
                })

        # Calculate final training loss and accuracy for epoch
        average_train_loss = accumulated_train_loss / len(train_data_loader)
        train_accuracy = 100. * running_correct_predictions / running_total_samples

        # Set the model to evaluation mode
        model.eval()
        test_loss = 0.0  # Track cumulative test loss
        correct_test_predictions = 0  # Track correct predictions for test accuracy
        total_test_samples = 0  # Track total samples in the test set

        # Evaluate model on test data
        with tqdm(test_data_loader, unit="batch", leave=False) as test_progress:
            test_progress.set_description(f"Epoch [{epoch + 1}/{num_epochs}] - Testing")
            with torch.no_grad():  # Disable gradient calculation for evaluation
                for batch_idx, (input_images, labels) in enumerate(test_progress):
                    input_images, labels = input_images.to(device), labels.to(device)

                    # Forward pass: Compute predictions on test data
                    predictions = model(input_images, dataset)
                    loss = loss_func(predictions, labels)
                    test_loss += loss.item()
                    # Calculate test accuracy
                    _, predicted_labels = torch.max(predictions, 1)
                    total_test_samples += labels.size(0)
                    correct_test_predictions += (predicted_labels == labels).sum().item()
                    # Update progress bar with running metrics
                    test_progress.set_postfix({
                        'Loss': test_loss / (batch_idx + 1),
                        'Accuracy': 100. * correct_test_predictions / total_test_samples
                    })

        # Calculate final test loss and accuracy for epoch
        average_test_loss = test_loss / len(test_data_loader)
        test_accuracy = 100. * correct_test_predictions / total_test_samples

        print(f"{epoch:>2}/{num_epochs:<1}{average_train_loss:>15f}{train_accuracy:>20f}{average_test_loss:>20f}{test_accuracy:>20f}") if \
            (num_epochs == 10 or (num_epochs == 50 and (epoch % 10 == 0 or epoch == num_epochs - 1))) else None

        # Step the learning rate scheduler based on test loss
        lr_scheduler.step(average_test_loss)

        # Save the model if test accuracy improves
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_filepath)

    print(f"Model trained on the {dataset} dataset with best test accuracy: {best_test_accuracy:.2f}% saved in file: {best_model_filepath}.")


# -------------------- Dataset Utilities --------------------
def get_dataset_info(dataset):
    """
    Retrieves dataset information such as number of input channels, number of classes,
    mean and standard deviation for normalization, and class labels

    Parameters:
        - dataset (str): Dataset for which to retrieve the information

    Returns:
        - input_channels (int): Number of input channels in the images (1 for MNIST, 3 for CIFAR-10)
        - num_classes (int): Number of target classes (10 for MNIST and CIFAR-10)
        - means (float or tuple): Mean values for normalization
        - stds (float or tuple):  Standard deviation values for normalization
        - labels (list): Dataset class labels
    """
    if dataset == 'cifar':
        cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        labels = cifar_dataset.classes  # Class labels
        input_channels = cifar_dataset[0][0].shape[0]  # Number of input channels (3 for RGB)
        num_classes = len(cifar_dataset.classes)  # Number of classes
        means = (cifar_dataset.data / 255.0).mean(axis=(0, 1, 2))  # Compute mean for normalization
        stds = (cifar_dataset.data / 255.0).std(axis=(0, 1, 2))  # Compute standard deviation for normalization
        return input_channels, num_classes, means, stds, labels

    elif dataset == 'mnist':
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        labels = mnist_dataset.classes  # Class labels
        input_channels = 1  # Number of input channels (1 for grayscale)
        num_classes = len(mnist_dataset.classes)  # Number of classes
        means = (mnist_dataset.data / 255.0).mean()  # Compute mean for normalization
        stds = (mnist_dataset.data / 255.0).std()  # Compute standard deviation for normalization
        return input_channels, num_classes, means, stds, labels


def load_dataset(dataset, batch_size, means, stds):
    """
    Prepares and loads the dataset with transformations for training and testing, including data augmentation,
    normalization, and batching

    Parameters:
        - dataset (str): Dataset to load
        - batch_size (int): Number of samples per batch to load
        - means (float or tuple): Mean values for normalization
        - stds (float or tuple): Standard deviation values for normalization

    Returns:
        - Tuple[DataLoader, DataLoader]: Train and test DataLoader objects for the dataset
    """
    if dataset == 'cifar':
        # Transformations for CIFAR-10 images:
        # 1. Randomly flip the image horizontally to introduce features from both orientations, helping model generalize better
        # 2. Randomly crop the image with padding to introduce varied object positions, helping model generalize better
        # 3. Randomly adjust brightness, contrast, saturation, and hue to help the model adapt to lighting and color variations
        # 4. Convert image to pytorch tensor
        # 5. Normalize image using the mean and standard deviation
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        # 1. Convert image to pytorch tensor
        # 2. Normalize image using the mean and standard deviation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        # Load datasets with transformation pipeline applied
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'mnist':
        # Transformation for MNIST images
        # 1. Convert image to pytorch tensor
        # 2. Normalize image using the mean and standard deviation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(means,), std=(stds,))
        ])
        # 1. Convert image to pytorch tensor
        # 2. Normalize image using the mean and standard deviation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(means,), std=(stds,))
        ])

        # Load datasets with transformation pipeline applied
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    # Load datasets into pytorch DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    return train_loader, test_loader


# -------------------- Image Utilities --------------------
def preprocess_image(input_image, dataset, means, stds):
    """
    Preprocess the input image for inference depending on the dataset. Resizes the image, normalizes it using the
    dataset specific mean and std,and converts to pytorch tensor

    Parameters:
        - input_image (str): Inference image file
        - dataset (str): Dataset the model is trained on
        - means (float or tuple): Mean values for normalization
        - stds (float or tuple): Standard deviation values for normalization

    Returns:
        - torch.Tensor: The preprocessed image tensor ready for input into the model
    """
    if dataset == 'mnist':
        # Convert the image to grayscale
        image = Image.open(input_image).convert("L")
        # Convert the image to np array to inspect brightness
        # MNIST images have black backgrounds and white digits
        image_np = np.array(image)
        # Calculate the mean brightness to determine if the image needs inversion
        # If the background is bright, image is inverted so the background is dark and the digit is bright
        mean_brightness = image_np.mean()
        if mean_brightness > 127:
            image = ImageOps.invert(image)
        # Transformation for MNIST images
        # 1. Resize image to 28x28 pixels (standard MNIST size)
        # 2. Convert image to pytorch tensor
        # 3. Normalize image using the mean and std
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((means,), (stds,))
        ])
    elif dataset == 'cifar':
        # Convert the image to RGB
        image = Image.open(input_image)
        # If the image has an alpha channel (like png files), convert it to RGB by removing the alpha
        if image.mode != 'RGB':
            image = image.convert("RGB")
        # Transformation for CIFAR-10 images:
        # 1. Resize image to 32x32 pixels (standard CIFAR-10 size)
        # 2. Convert image to pytorch tensor
        # 3. Normalize image using the mean and std
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

    # Apply the transformation pipeline to the input image
    image = transform(image).unsqueeze(0)
    return image


# -------------------- Inference --------------------
def classify_image(model, input_image, device, dataset, mean, std, dataset_labels):
    """
    Perform inference on a single image to classify it using the trained model, and visualize the feature maps from the first
    convolutional layer

    Parameters:
        - model: Trained CNN model for classification
        - input_image (str): Inference image file
        - device: Device (CPU/GPU) on which to run the inference
        - dataset (str): Dataset the model is trained on
        - means (float or tuple): Mean values for normalization
        - stds (float or tuple): Standard deviation values for normalization
        - dataset_labels (list): List of class labels specific to the dataset

    Returns:
        - str: Predicted class label of the input image
       """

    # Preprocess input image: Resize, normalize, and convert to tensor
    image = preprocess_image(input_image, dataset, mean, std).to(device)

    # Set model to evaluation mode
    model.eval()
    # Perform inference without tracking gradients
    with torch.no_grad():
        # Pass image through the first convolutional layer to extract feature maps
        conv1_output = model.C1(image)
        # Forward pass
        output = model(image, dataset)
        # Predicted class index
        _, predicted = torch.max(output, 1)
        predicted_class_idx = predicted.item()
    # Visualize feature maps of first convolutional layer
    visualize_first_conv_layer(conv1_output, dataset)
    # predicted class label
    return dataset_labels[predicted_class_idx]


# -------------------- Visualization --------------------
def visualize_first_conv_layer(conv1_output, dataset):
    """
    Visualizes the feature maps (filters) from the first convolutional layer of the CNN model

    Parameters:
        - conv1_output (torch.Tensor): The output of the first convolutional layer
        - dataset (str): Dataset the model is trained on

    Returns:
        - None. Saves file with visualized feature maps
    """

    # Move the tensor to CPU for visualization in matplotlib
    conv1_output = conv1_output.cpu()
    # Number of filters (channels) in the first convolutional layer
    num_filters = conv1_output.shape[1]
    # Grid of subplots to display 32 feature maps
    fig, axes = plt.subplots(4, 8, figsize=(14, 8), dpi=300)

    for i in range(num_filters):
        # Determine the position of the current filter in the grid (row, col)
        ax = axes[i // 8, i % 8]
        # Convert the feature map to a numpy array for visualization
        feature_map = conv1_output[0, i].numpy()
        # Normalize the feature map for better visual clarity (scale to range 0-1)
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        # Display the feature map on the subplot with grayscale color map and bicubic interpolation for smoothness
        ax.imshow(feature_map, cmap='gray', interpolation='bicubic')
        ax.axis('off')

    output_filename = f"CONV_rslt_{dataset}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()


# -------------------- Inference utilities --------------------

def load_saved_model(model_class, num_classes, in_channels, model_directory, dataset, device):
    """
    Loads a saved model and prepares it for inference.

    Parameters:
        - model_class: Model class to instantiate (CNNClassifier).
        - num_classes (int): Number of output classes for classification.
        - in_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR-10).
        - model_directory (str): Directory where the saved model is stored.
        - dataset (str): Dataset the model is trained on.
        - device: Device (CPU/GPU) on which to load model on for inference.

        Returns:
            torch.nn.Module: The trained model loaded with saved state, ready for inference.
        """
    # Construct path for saved model
    model_filename = f"{dataset}_trained_model.pth"
    model_path = os.path.join(model_directory, model_filename)

    # Check if model file exists
    if not os.path.exists(model_path):
        return None

    # Initialize model with the parameters
    model = model_class(in_channels, num_classes, dataset)
    # Load model saved state (weights and biases)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Move model to the specified device (CPU/GPU)
    model.to(device)
    # Set model to evaluation mode
    model.eval()
    return model

# -------------------- Evaluation --------------------
def evaluate_model(model, data_loader, device, dataset_name, num_classes, k_values=[1, 5]):
    """
    Perform post-training analysis on a trained model.

    Parameters:
        - model (torch.nn.Module): Trained CNN model.
        - data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        - device (torch.device): Device to run the evaluation on (CPU/GPU).
        - dataset_name (str): Name of the dataset for analysis.
        - num_classes (int): Number of target classes.
        - k_values (list): List of 'k' values for Top-K accuracy (e.g., [1, 5]).

    Returns:
        - None (prints analysis results to the console).
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    top_k_correct = {k: 0 for k in k_values}

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, dataset_name)
            
            # Predicted probabilities and top-K predictions
            probs = torch.softmax(outputs, dim=1)
            top_k_preds = {k: probs.topk(k, dim=1).indices for k in k_values}
            
            # Update top-K accuracy counts
            for k in k_values:
                top_k_correct[k] += (
                    torch.eq(labels.view(-1, 1), top_k_preds[k]).any(dim=1).sum().item()
                )

            # Store predictions and labels for further analysis
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert lists to NumPy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metrics calculation
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

# -------------------- CLI --------------------

def main():
    # CLI Usage examples
    epilog = """Usage examples:
      python3 CNNclassify.py train --mnist       Train the model using the MNIST dataset
      python3 CNNclassify.py train --cifar       Train the model using the CIFAR-10 dataset
      python3 CNNclassify.py test car.png        Test the model using 'car.png'
    """

    # Set up the argument parser for CLI commands
    parser = argparse.ArgumentParser(
        usage='python3 CNNclassify.py [-h] {train,test,save} ...',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        epilog=epilog
    )

    # Create subparsers for train and test commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute: train, test, or save")

    # Parser for the train command
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--mnist', action='store_true', help="Train the CNN using the MNIST dataset.")
    train_parser.add_argument('--cifar', action='store_true', help="Train the CNN using the CIFAR-10 dataset.")

    # Parser for the test command
    test_parser = subparsers.add_parser('test', help="Test the model")
    test_parser.add_argument('image_file', nargs='*', help="Image file(s) for testing (e.g., car.png)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # If no arguments or no command is given, show help and exit
    if len(sys.argv) == 1 or args.command is None:
        parser.print_help()
        sys.exit(1)

    # Train the model based on user input
    if args.command == 'train':
        # If no dataset is provided for training, show an error and exit
        if not args.mnist and not args.cifar:
            print("Error: 'train' command requires either --mnist or --cifar argument.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        # Set up the device (GPU/CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Choose the dataset to train on (MNIST or CIFAR-10)
        dataset = 'mnist' if args.mnist else 'cifar'

        # Get dataset-specific information (channels, classes, mean, std, and labels)
        batch_size = 64
        input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
        # Load the dataset into DataLoader for batching
        train_loader, test_loader = load_dataset(dataset, batch_size, means, stds)
        # Initialize the CNN model with appropriate input channels and number of classes
        model = CNNClassifier(input_channels, num_classes, dataset).to(device)
        # Set up the loss function, optimizer and learning rate scheduler for training
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)
        # Set the number of training epochs (10 for MNIST, 50 for CIFAR-10)
        num_epochs = 10 if dataset == 'mnist' else 50
        # Train the model
        train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, device, num_epochs, dataset)

    # Test the model using an image file
    elif args.command == 'test':
        if not args.image_file:
            print("Error: 'test' command requires at least one image file.", file=sys.stderr)
            print("Use --help for more information.")
            sys.exit(1)

        # Set the model directory and device
        model_dir = "model"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for img_file in args.image_file:
            # List of datasets to run inference on
            datasets = ['cifar', 'mnist']
            # Run inference using models trained on both CIFAR-10 and MNIST
            for dataset in datasets:
                # Get dataset-specific details (channels, classes, mean, std, labels)
                input_channels, num_classes, means, stds, dataset_labels = get_dataset_info(dataset)
                # Load the saved model for the current dataset
                model = load_saved_model(CNNClassifier, num_classes, input_channels, model_dir, dataset, device)
                if model is not None:
                    # Perform inference on the image and get the predicted label
                    predicted_label = classify_image(model, img_file, device, dataset, means, stds, dataset_labels)
                    if dataset == 'mnist':
                        print(f"Prediction result by model trained on MNIST dataset: {predicted_label}")
                    elif dataset == 'cifar':
                        print(f"Prediction result by model trained on CIFAR-10 dataset: {predicted_label}")
                else:
                    print(f"Error: Could not load model for {dataset} dataset.")


if __name__ == '__main__':
    main()
