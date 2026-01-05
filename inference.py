"""
Inference script for Architectural Style Recognition.

Takes an input folder of images and outputs a CSV with probabilities for all 25 styles.

Usage:
    python inference.py --input_folder ./my_images --output results.csv
    python inference.py --input_folder ./my_images --output results.csv --model_path ./best_model.pth

Output CSV format:
    file, Achaemenid_architecture_prob, American_Foursquare_architecture_prob, ..., Tudor_Revival_architecture_prob
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# 25 architectural style classes (sorted alphabetically as used in training)
CLASSES = [
    "Achaemenid architecture",
    "American Foursquare architecture",
    "American craftsman style",
    "Ancient Egyptian architecture",
    "Art Deco architecture",
    "Art Nouveau architecture",
    "Baroque architecture",
    "Bauhaus architecture",
    "Beaux-Arts architecture",
    "Byzantine architecture",
    "Chicago school architecture",
    "Colonial architecture",
    "Deconstructivism",
    "Edwardian architecture",
    "Georgian architecture",
    "Gothic architecture",
    "Greek Revival architecture",
    "International style",
    "Novelty architecture",
    "Palladian architecture",
    "Postmodern architecture",
    "Queen Anne architecture",
    "Romanesque architecture",
    "Russian Revival architecture",
    "Tudor Revival architecture",
]

# Column names for CSV (convert to valid column names)
COLUMN_NAMES = [c.replace(" ", "_").replace("-", "_") + "_prob" for c in CLASSES]

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 299


def load_model(model_path: str, device: str = "cuda"):
    """Load the trained EfficientNetV2-L model."""
    print(f"Loading model from {model_path}...")
    
    # Create model architecture
    model = models.efficientnet_v2_l(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully ({len(CLASSES)} classes)")
    return model


def get_transform():
    """Get inference transform."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def predict_image(model, image_path: Path, transform, device: str):
    """Predict architectural style probabilities for a single image."""
    try:
        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        return probabilities.cpu().numpy()
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def find_images(input_folder: Path):
    """Find all images in the input folder (recursive)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    images = []
    
    for ext in image_extensions:
        images.extend(input_folder.rglob(f"*{ext}"))
        images.extend(input_folder.rglob(f"*{ext.upper()}"))
    
    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(description="Architectural Style Recognition Inference")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to folder containing images")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV file path (default: predictions.csv)")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="Path to model weights (default: best_model.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu, default: cuda)")
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Find images
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        return
    
    images = find_images(input_folder)
    if not images:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(images)} images in {input_folder}")
    
    # Load model
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        print("\nTo download the best model:")
        print("  1. Download from Google Drive link in README.md")
        print("  2. Or run: modal volume get archstyle-sweep-results / ./sweep_results")
        return
    
    model = load_model(args.model_path, args.device)
    transform = get_transform()
    
    # Process images
    results = []
    for image_path in tqdm(images, desc="Processing"):
        probs = predict_image(model, image_path, transform, args.device)
        
        if probs is not None:
            result = {"file": str(image_path.relative_to(input_folder))}
            
            # Add probability for each class
            for i, col_name in enumerate(COLUMN_NAMES):
                result[col_name] = f"{probs[i]:.6f}"
            
            results.append(result)
    
    # Write CSV
    output_path = Path(args.output)
    fieldnames = ["file"] + COLUMN_NAMES
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(results)} images")
    print(f"Columns: file + {len(COLUMN_NAMES)} style probabilities")
    
    # Print sample results (top prediction for first 5 images)
    if results:
        print("\nSample predictions (showing top style for each):")
        for r in results[:5]:
            # Find top prediction
            max_prob = 0
            max_style = ""
            for col in COLUMN_NAMES:
                prob = float(r[col])
                if prob > max_prob:
                    max_prob = prob
                    max_style = col.replace("_prob", "").replace("_", " ")
            print(f"  {r['file']}: {max_style} ({max_prob:.4f})")


if __name__ == "__main__":
    main()
