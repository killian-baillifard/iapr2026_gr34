import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from project.scripts.preprocessing.cache import load_preprocessing_cache

# 1. Import the model architecture (must match the training code)
from cnn import UNOCNNClassifier

MODEL_PATH = os.path.join("best_model.pth")

def run_inference():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load the data
    print("Loading preprocessed data...")
    images, labels = load_preprocessing_cache()
    
    # Select a random index
    idx = random.randint(0, len(images) - 1)
    image_path = images[idx]
    true_label = labels[idx]

    # 3. Preprocess the image (Exact same steps as UNOSectorizedDataset)
    image_np = np.load(image_path)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
    
    # Resize to the input size the model expects
    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0), 
        size=(256, 512), 
        mode="bilinear", 
        align_corners=False
    ).to(device)

    # 4. Load Model and Weights
    model = UNOCNNClassifier(num_classes=54).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please train the model first.")
        return

    # 5. Inference
    with torch.no_grad():
        logits = model(image_tensor)
        # Apply Sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # 6. Display Results
    print(f"\n--- Results for sample: {image_path} ---")
    
    # Print indices where probability is > 0.5 (detected cards)
    detected_indices = np.where(probs > 0.5)[0]
    actual_indices = np.where(true_label == 1)[0]

    print(f"Actual Card Indices:   {actual_indices}")
    print(f"Detected Card Indices: {detected_indices}")
    
    # Show the top 5 highest confidence scores
    print("\nTop 5 Probabilities:")
    top_5_idx = np.argsort(probs)[-5:][::-1]
    for i in top_5_idx:
        print(f"  Class {i:02d}: {probs[i]:.4f}")

if __name__ == "__main__":
    run_inference()
