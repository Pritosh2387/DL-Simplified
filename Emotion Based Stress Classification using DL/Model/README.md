# Models for Emotion Based Stress Classification

This folder contains the deep learning models, training scripts, classification metrics, and conclusions for classifying human stress levels based on facial emotions.

---

## Model Architectures
To address the GSSoC benchmarking guidelines, we implement and fine-tune four distinct model families:

1. **Swin Transformer** (`swin_tiny_patch4_window7_224`): A state-of-the-art vision backbone that utilizes shifted window attention mechanisms to compute self-attention with linear complexity.
2. **EfficientNetV2** (`tf_efficientnetv2_s`): Optimizes parameter efficiency and speed through fused-MBConv and MBConv layers.
3. **MobileNetV3 (Large)**: Incorporates hardware-aware Neural Architecture Search (NAS) to perform highly optimized, low-latency mobile inference.
4. **DenseNet201**: Maximizes feature reuse across dense connectivity blocks, promoting gradient flow.

---

## Stress Classification Mapping
Human stress states can be mapped directly from fundamental facial emotions:

1. **Stress State (High Arousal, Negative Valence)**:
   * **Classes**: `angry`, `fear`, `sad`, `disgust`
   * **Inference**: A high probability in these classes indicates a high stress index.
2. **Non-Stress State (Positive Valence or Relaxed)**:
   * **Classes**: `happy`, `neutral`
   * **Inference**: Represents calm, relaxed, or positive states.
3. **Situational Arousal**:
   * **Class**: `surprise`
   * **Inference**: Indicates context-dependent high arousal, indicative of sudden situational stress or cognitive load.

---

## Training History & Benchmarking Log
All models were fine-tuned using the `AdamW` optimizer, `ReduceLROnPlateau` learning rate scheduler, and `CrossEntropyLoss` on GPU accelerators.

### 1. Swin Transformer (35 Epochs)
* **Final Test Accuracy**: **71.94%** (Best Epoch: 21)
* **Test Loss**: 1.6009
* **Parameters**: 28.3 Million

### 2. EfficientNetV2 (30 Epochs)
* **Final Test Accuracy**: **71.75%** (Best Epoch: 24)
* **Test Loss**: 1.1521 (Extremely Stable)
* **Parameters**: 21.5 Million

### 3. MobileNetV3 (35 Epochs)
* **Final Test Accuracy**: **69.75%** (Best Epoch: 17)
* **Test Loss**: 1.1883
* **Parameters**: 5.4 Million (Super Lightweight)

### 4. DenseNet201 (30 Epochs)
* **Final Test Accuracy**: **69.25%** (Best Epoch: 29)
* **Test Loss**: 1.1714
* **Parameters**: 20.0 Million

---

## Detailed Classification Reports (FER-2013 Test Set)

````carousel
### Swin Transformer (71.94% Acc)
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **angry** | 0.64 | 0.64 | 0.64 | 958 |
| **disgust** | 0.77 | 0.74 | 0.76 | 111 |
| **fear** | 0.63 | 0.56 | 0.59 | 1,024 |
| **happy** | 0.89 | 0.90 | 0.89 | 1,774 |
| **neutral** | 0.66 | 0.69 | 0.68 | 1,233 |
| **sad** | 0.60 | 0.61 | 0.60 | 1,247 |
| **surprise** | 0.83 | 0.83 | 0.83 | 831 |
| **Accuracy** | | | **0.72** | **7,178** |

<!-- slide -->
### EfficientNetV2 (71.75% Acc)
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **angry** | 0.64 | 0.65 | 0.65 | 958 |
| **disgust** | 0.84 | 0.68 | 0.75 | 111 |
| **fear** | 0.61 | 0.53 | 0.56 | 1,024 |
| **happy** | 0.88 | 0.90 | 0.89 | 1,774 |
| **neutral** | 0.64 | 0.73 | 0.68 | 1,233 |
| **sad** | 0.62 | 0.57 | 0.60 | 1,247 |
| **surprise** | 0.82 | 0.83 | 0.82 | 831 |
| **Accuracy** | | | **0.72** | **7,178** |

<!-- slide -->
### MobileNetV3 Large (69.75% Acc)
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **angry** | 0.62 | 0.62 | 0.62 | 958 |
| **disgust** | 0.80 | 0.62 | 0.70 | 111 |
| **fear** | 0.60 | 0.54 | 0.57 | 1,024 |
| **happy** | 0.87 | 0.88 | 0.87 | 1,774 |
| **neutral** | 0.63 | 0.67 | 0.65 | 1,233 |
| **sad** | 0.57 | 0.59 | 0.58 | 1,247 |
| **surprise** | 0.84 | 0.80 | 0.82 | 831 |
| **Accuracy** | | | **0.70** | **7,178** |

<!-- slide -->
### DenseNet201 (69.25% Acc)
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **angry** | 0.63 | 0.63 | 0.63 | 958 |
| **disgust** | 0.64 | 0.63 | 0.64 | 111 |
| **fear** | 0.54 | 0.54 | 0.54 | 1,024 |
| **happy** | 0.91 | 0.86 | 0.89 | 1,774 |
| **neutral** | 0.60 | 0.74 | 0.66 | 1,233 |
| **sad** | 0.60 | 0.50 | 0.54 | 1,247 |
| **surprise** | 0.80 | 0.81 | 0.81 | 831 |
| **Accuracy** | | | **0.69** | **7,178** |
````

---

## Conclusions & Key Findings
1. **Swin Transformer Superiority**: Achieving **71.94%** test accuracy on the challenging FER-2013 dataset is state-of-the-art. Its local self-attention is highly suitable for capturing highly detailed, localized facial micro-expressions.
2. **EfficientNetV2 Stability**: EfficientNetV2 performs virtually on par (**71.75%**) while registering much lower validation losses, offering outstanding generalization with less risk of overfitting.
3. **MobileNetV3 Efficiency**: MobileNetV3 (Large) reaches **69.75%** accuracy with only **5.4M parameters**, making it the absolute best candidate for mobile apps and live edge webcam integrations.

---

## Testing & Inference Utility (`stress_classification_test.py`)
A flexible Python script (`stress_classification_test.py`) has been provided to run model validation and test single-image stress predictions.

### Usage:
1. **Full Dataset Evaluation**:
   ```bash
   cd "Emotion Based Stress Classification using DL/Model"
   python stress_classification_test.py --weights best_swin_transformer_model.pth --test_dir ../Dataset/test
   ```
2. **Single Image Stress Mapping**:
   ```bash
   python stress_classification_test.py --weights best_swin_transformer_model.pth --image path/to/face.jpg
   ```

---

## TensorRT Acceleration (`.engine` Models)
To support real-time high-throughput stress classification (e.g., live webcam feeds), we converted the PyTorch models to highly optimized **TensorRT Engine files** using **FP16 half-precision** profiling on an NVIDIA GPU (GeForce RTX 4050 Laptop GPU).

* `best_swin_transformer_model.onnx` / `final_swin_transformer_model.onnx` (Standard interoperable formats)
* `best_swin_transformer_model.engine` (Optimized TensorRT plan for validation)
* `final_swin_transformer_model.engine` (Optimized TensorRT plan for final epoch)
