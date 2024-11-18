# Adaptive CNN Classification Pipeline

A modular and extensible pipeline for image classification using custom CNN architectures, designed to handle datasets with diverse characteristics such as MNIST (grayscale, single-channel) and CIFAR-10 (color, multi-channel). The pipeline includes:

- **Data Handling**: Automated dataset loading, preprocessing, augmentation, and normalization tailored to dataset-specific requirements.
- **Dynamic Model Architectures**: Custom CNNs with varying depths and configurations optimized for each dataset.
- **Training Workflow**: Integrated support for gradient clipping, learning rate scheduling (ReduceLROnPlateau), and model checkpointing based on test accuracy.
- **Evaluation and Inference**: Tools for testing accuracy, loss tracking, and feature visualization, enabling interpretability of CNN activations.
- **Visualization**: Generation of feature map heatmaps from the first convolutional layer to explore learned patterns.
- **CLI**: Flexible options for training and testing, supporting deployment-ready model inference on new images.

### **Classification Metrics**

#### **CIFAR-10**
- **Precision (Macro):** 0.8650  
- **Recall (Macro):**    0.8656  
- **F1 Score (Macro):**  0.8651  

---

### **Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.86      | 0.90   | 0.88     | 1000    |
| 1     | 0.93      | 0.96   | 0.94     | 1000    |
| 2     | 0.81      | 0.81   | 0.81     | 1000    |
| 3     | 0.77      | 0.71   | 0.74     | 1000    |
| 4     | 0.83      | 0.85   | 0.84     | 1000    |
| 5     | 0.80      | 0.79   | 0.79     | 1000    |
| 6     | 0.89      | 0.92   | 0.90     | 1000    |
| 7     | 0.91      | 0.89   | 0.90     | 1000    |
| 8     | 0.93      | 0.91   | 0.92     | 1000    |
| 9     | 0.92      | 0.91   | 0.92     | 1000    |

**Overall:**
- **Accuracy:** 0.87  
- **Macro Avg:** Precision = 0.87, Recall = 0.87, F1-Score = 0.87  
- **Weighted Avg:** Precision = 0.87, Recall = 0.87, F1-Score = 0.87  


#### **MNIST**
- **Precision (Macro):** 0.9905  
- **Recall (Macro):**    0.9906  
- **F1 Score (Macro):**  0.9905  

---

### **Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 1.00   | 0.99     | 980     |
| 1     | 0.99      | 0.99   | 0.99     | 1135    |
| 2     | 0.99      | 0.99   | 0.99     | 1032    |
| 3     | 0.99      | 0.99   | 0.99     | 1010    |
| 4     | 0.99      | 0.99   | 0.99     | 982     |
| 5     | 0.99      | 1.00   | 0.99     | 892     |
| 6     | 0.99      | 0.99   | 0.99     | 958     |
| 7     | 1.00      | 0.98   | 0.99     | 1028    |
| 8     | 0.98      | 1.00   | 0.99     | 974     |
| 9     | 0.98      | 0.99   | 0.99     | 1009    |

**Overall:**
- **Accuracy:** 0.99  
- **Macro Avg:** Precision = 0.99, Recall = 0.99, F1-Score = 0.99  
- **Weighted Avg:** Precision = 0.99, Recall = 0.99, F1-Score = 0.99
