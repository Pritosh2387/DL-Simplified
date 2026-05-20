# =====================================================================
# Emotion-Based Stress Classification - Testing and Inference Utility
# Model: Swin Transformer (swin_tiny_patch4_window7_224)
# Framework: PyTorch
# =====================================================================

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import timm

# =========================
# Configuration & Constants
# =========================
DEFAULT_TEST_DIR = os.path.join("..", "Dataset", "test")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_swin_transformer_model.pth")
DEFAULT_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Stress Classification Mapping
STRESS_MAPPING = {
    'angry': 'Stress State (High)',
    'disgust': 'Stress State (High)',
    'fear': 'Stress State (High)',
    'sad': 'Stress State (High)',
    'surprise': 'Transient Arousal (Context-Dependent)',
    'happy': 'Non-Stress State (Relaxed)',
    'neutral': 'Non-Stress State (Calm)'
}

# Image Preprocessing Transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =========================
# TensorRT Engine Wrapper
# =========================
class TRTEngineWrapper:
    """
    Drop-in replacement for a PyTorch module to run TensorRT engine (.engine) 
    inference directly using PyTorch CUDA memory pointers.
    """
    def __init__(self, engine_path, device="cuda"):
        self.device = device
        print(f"[*] Loading TensorRT engine from: {engine_path}...")
        
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "[!] TensorRT Python library is not installed. "
                "Please run 'pip install tensorrt' to run .engine models."
            )
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
            
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
    def __call__(self, images):
        batch_size = images.size(0)
        
        # Ensure images are on CUDA
        if not images.is_cuda:
            images = images.to('cuda')
            
        # Ensure float32 dtype
        if images.dtype != torch.float32:
            images = images.float()
            
        # Allocate output tensor directly on CUDA
        output_tensor = torch.empty((batch_size, 7), dtype=torch.float32, device='cuda')
        
        # Bind GPU addresses and shape
        self.context.set_input_shape("input", (batch_size, 3, 224, 224))
        self.context.set_tensor_address("input", images.data_ptr())
        self.context.set_tensor_address("output", output_tensor.data_ptr())
        
        # Execute asynchronously using PyTorch CUDA Stream
        with torch.cuda.stream(self.stream):
            success = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            if not success:
                raise RuntimeError("[!] TensorRT execution failed.")
                
        # Synchronize PyTorch stream
        self.stream.synchronize()
        return output_tensor

    def eval(self):
        # No-op for interface compatibility
        pass

    def to(self, device):
        # No-op for interface compatibility
        return self


# =========================
# Load Model Function
# =========================
def load_swin_model(weights_path, device, num_classes=7):
    """
    Initializes the Swin Transformer model (or TensorRT engine) and loads weights.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[!] Weights file not found at: {weights_path}")
        
    # Check if target is a TensorRT Engine
    if weights_path.endswith(".engine"):
        if "cuda" not in str(device):
            raise ValueError(
                "[!] TensorRT engines can only run on CUDA-enabled GPU devices. "
                "Please configure PyTorch to run on CUDA (gpu) or load a .pth file on CPU instead."
            )
        return TRTEngineWrapper(weights_path, device)
        
    print(f"[*] Initializing Swin Transformer architecture...")
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=num_classes
    )
    
    print(f"[*] Loading trained weights from: {weights_path}...")
    # Load state dict and handle CPU/GPU mapping
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# =========================
# Dataset Evaluation
# =========================
def evaluate_dataset(model, test_dir, device):
    """
    Runs evaluation over the entire test dataset directory and prints metrics.
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"[!] Test dataset directory not found at: {test_dir}")
        
    print(f"[*] Loading test dataset from: {test_dir}...")
    test_dataset = ImageFolder(root=test_dir, transform=image_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    class_names = test_dataset.classes
    
    print(f"[*] Evaluating {len(test_dataset)} images...")
    predictions = []
    actuals = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    print("\n" + "="*50)
    print(f"                     RESULTS")
    print("="*50)
    print(f"Average Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy:   {test_acc:.2f}%")
    print("="*50)
    
    print("\n[+] Classification Report:\n")
    print(classification_report(actuals, predictions, target_names=class_names))
    
    # Save Confusion Matrix
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.tight_layout()
    
    cm_path = "confusion_matrix_test.png"
    plt.savefig(cm_path)
    print(f"[+] Confusion matrix plot saved locally as: {cm_path}")

# =========================
# Single Image Inference
# =========================
def predict_single_image(model, image_path, device, class_names=DEFAULT_CLASSES):
    """
    Runs prediction on a single facial image and maps it to a stress level.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[!] Target image not found at: {image_path}")
        
    print(f"[*] Preprocessing target image: {image_path}...")
    img = Image.open(image_path).convert('RGB')
    input_tensor = image_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    probabilities = probabilities.cpu().numpy()
    pred_idx = np.argmax(probabilities)
    pred_emotion = class_names[pred_idx]
    confidence = probabilities[pred_idx] * 100
    stress_status = STRESS_MAPPING.get(pred_emotion, 'Unknown')
    
    print("\n" + "="*50)
    print(f"             SINGLE IMAGE INFERENCE")
    print("="*50)
    print(f"Predicted Emotion:    {pred_emotion.upper()}")
    print(f"Confidence Score:     {confidence:.2f}%")
    print(f"Stress Classification: {stress_status.upper()}")
    print("="*50)
    
    print("\n[+] Full Class Probability Breakdown:")
    for idx, prob in enumerate(probabilities):
        print(f" - {class_names[idx].ljust(10)}: {prob*100:6.2f}%  -> ({STRESS_MAPPING.get(class_names[idx], 'Unknown')})")
    print("="*50 + "\n")

# =========================
# Main Execution Entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Run Inference on trained Emotion Based Stress Classification models.")
    parser.add_argument("--weights", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model .pth weights.")
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR, help="Path to test dataset directory.")
    parser.add_argument("--image", type=str, default="", help="Path to a single image file for inference.")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    
    try:
        model = load_swin_model(args.weights, device)
        
        if args.image:
            # Mode 2: Single Image Inference
            predict_single_image(model, args.image, device)
        else:
            # Mode 1: Full Dataset Evaluation
            evaluate_dataset(model, args.test_dir, device)
            
    except Exception as e:
        print(f"\n[!] Error during execution:\n{str(e)}")
