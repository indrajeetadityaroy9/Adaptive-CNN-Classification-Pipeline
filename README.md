# Dataset Adaptive Image Classification

A modular CNN classification pipeline that dynamically adjusts its architecture based on dataset characteristics. The pipeline implements dataset-specific model configurations, preprocessing strategies, and augmentation techniques to optimize performance across diverse image classification tasks.

1. **Adaptive Architecture Design**: A single model class that dynamically instantiates dataset-specific layer configurations
2. **Automated Preprocessing**: Dataset-aware image preprocessing and augmentation pipelines
3. **Training Methods**: Integration of gradient clipping, adaptive learning rate scheduling, and checkpointing
4. **Interpretability Tools**: First-layer feature map visualization for understanding learned representations

### Architecture

**MNIST Configuration** (shallow architecture for simple grayscale data):
- Convolutional layers: 2 (1→32→64 channels)
- Kernel sizes: 5×5, 3×3
- Pooling: MaxPool2d (2×2) + AdaptiveAvgPool2d (4×4)
- Fully connected: 1024→128→10

**CIFAR-10 Configuration** (deeper architecture for complex color images):
- Convolutional layers: 4 (3→32→64→128→256 channels)
- Kernel sizes: 5×5, 3×3, 3×3, 3×3
- Pooling: MaxPool2d (2×2) + AdaptiveAvgPool2d (4×4)
- Fully connected: 4096→512→128→10

Both architectures employ:
- **Batch Normalization** after each convolutional layer for training stability
- **Dropout** (p=0.5) for regularization
- **ReLU activations** for non-linearity

### Dataset Preprocessing

**MNIST Pipeline**:
- Grayscale conversion with automatic brightness inversion (threshold: 127)
- Resize to 28×28 pixels
- Normalization: μ=0.1307, σ=0.3081

**CIFAR-10 Pipeline**:
- **Training augmentation**: Random horizontal flip, random crop (32×32, padding=4), color jitter (brightness, contrast, saturation, hue: ±0.2)
- **Test preprocessing**: Resize to 32×32 pixels
- Normalization: per-channel μ and σ computed from training set

### Training

**Optimization**:
- Optimizer: Adam (lr=0.001, weight_decay=0.0005)
- Loss function: Cross-entropy
- Gradient clipping: max value = 0.1 (prevents exploding gradients)
- Learning rate scheduler: ReduceLROnPlateau (factor=0.1, patience=2 epochs)

**Training Configuration**:
- Batch size: 64
- MNIST epochs: 10
- CIFAR-10 epochs: 50
- Checkpointing: Save best model based on test accuracy

## Experimental Setup

### Datasets

**MNIST** (Handwritten Digits):
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28×28 (grayscale)
- Classes: 10 (digits 0-9)

**CIFAR-10** (Natural Images):
- Training samples: 50,000
- Test samples: 10,000
- Image size: 32×32 (RGB)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Macro-averaged across all classes
- **Confusion Matrix**: Per-class error analysis
- **Top-K Accuracy**: Top-1 and Top-5 predictions

## Results

### MNIST Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.0%** |
| **Precision (Macro)** | 0.9905 |
| **Recall (Macro)** | 0.9906 |
| **F1-Score (Macro)** | 0.9905 |

### CIFAR-10 Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **87.0%** |
| **Precision (Macro)** | 0.8650 |
| **Recall (Macro)** | 0.8656 |
| **F1-Score (Macro)** | 0.8651 |
