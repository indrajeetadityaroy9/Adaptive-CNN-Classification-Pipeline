# Adaptive CNN Classification Pipeline

A modular and extensible pipeline for image classification using custom CNN architectures, designed to handle datasets with diverse characteristics such as MNIST (grayscale, single-channel) and CIFAR-10 (color, multi-channel). The pipeline includes:

- **Data Handling**: Automated dataset loading, preprocessing, augmentation, and normalization tailored to dataset-specific requirements.
- **Dynamic Model Architectures**: Custom CNNs with varying depths and configurations optimized for each dataset.
- **Training Workflow**: Integrated support for gradient clipping, learning rate scheduling (ReduceLROnPlateau), and model checkpointing based on test accuracy.
- **Evaluation and Inference**: Tools for testing accuracy, loss tracking, and feature visualization, enabling interpretability of CNN activations.
- **Visualization**: Generation of feature map heatmaps from the first convolutional layer to explore learned patterns.
- **CLI**: Flexible options for training and testing, supporting deployment-ready model inference on new images.
