"""
Evaluate model performance on hybrid architectural style buildings.

This script tests the model's ability to recognize buildings that intentionally
combine multiple architectural styles, evaluating whether the model assigns
high probabilities to all relevant styles.

Usage:
    python evaluate_hybrids.py --model_path best_model.pth
    python evaluate_hybrids.py --model_path best_model.pth --output_dir hybrid_results

Metrics:
    - All Correct (top-k): Are all ground truth styles in the top-k predictions?
    - Any Correct (top-k): Is at least one ground truth style in the top-k?
    - Mean Probability: Average probability assigned to correct styles
    - Mean Reciprocal Rank: How highly ranked are the correct styles?
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


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

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 299


def load_model(model_path: str, device: str = "cuda"):
    """Load the trained EfficientNetV2-L model."""
    print(f"Loading model from {model_path}...")
    
    model = models.efficientnet_v2_l(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    
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
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        return probabilities.cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_hybrid_buildings(csv_path: Path, images_dir: Path):
    """Load hybrid building definitions from CSV."""
    buildings = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get ground truth styles
            styles = [row['style_1'], row['style_2']]
            if row.get('style_3') and row['style_3'].strip():
                styles.append(row['style_3'])
            
            # Get image paths
            images = []
            for file_col in ['file_1', 'file_2']:
                if row.get(file_col):
                    img_path = images_dir / row[file_col]
                    if img_path.exists():
                        images.append(img_path)
            
            if images and styles:
                buildings.append({
                    'name': row['building'],
                    'styles': styles,
                    'num_styles': len(styles),
                    'images': images,
                    'explanation': row.get('explanation', ''),
                })
    
    return buildings


def style_to_class_idx(style_name: str) -> int:
    """Convert style name to class index."""
    # Try exact match first
    if style_name in CLASSES:
        return CLASSES.index(style_name)
    
    # Try case-insensitive match
    style_lower = style_name.lower()
    for i, cls in enumerate(CLASSES):
        if cls.lower() == style_lower:
            return i
    
    # Try partial match
    for i, cls in enumerate(CLASSES):
        if style_lower in cls.lower() or cls.lower() in style_lower:
            return i
    
    return -1  # Not found


def evaluate_building(building: dict, model, transform, device: str) -> dict:
    """Evaluate model performance on a single hybrid building."""
    
    # Get ground truth class indices
    gt_indices = []
    for style in building['styles']:
        idx = style_to_class_idx(style)
        if idx >= 0:
            gt_indices.append(idx)
    
    if not gt_indices:
        print(f"Warning: No valid style indices for {building['name']}")
        return None
    
    k = len(gt_indices)  # Number of styles to check in top-k
    
    # Collect predictions for all images
    all_probs = []
    for img_path in building['images']:
        probs = predict_image(model, img_path, transform, device)
        if probs is not None:
            all_probs.append(probs)
    
    if not all_probs:
        return None
    
    # Average probabilities across images
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get top-k predictions
    top_k_indices = np.argsort(avg_probs)[::-1][:k]
    top_k_probs = avg_probs[top_k_indices]
    
    # Calculate metrics
    # 1. All Correct: Are all ground truth styles in top-k?
    all_correct = all(idx in top_k_indices for idx in gt_indices)
    
    # 2. Any Correct: Is at least one ground truth style in top-k?
    any_correct = any(idx in top_k_indices for idx in gt_indices)
    
    # 3. Count of correct styles in top-k
    num_correct = sum(1 for idx in gt_indices if idx in top_k_indices)
    pct_correct = num_correct / len(gt_indices)
    
    # 4. Mean probability assigned to correct styles
    correct_probs = [avg_probs[idx] for idx in gt_indices]
    mean_correct_prob = np.mean(correct_probs)
    
    # 5. Mean Reciprocal Rank (MRR) of correct styles
    ranks = []
    sorted_indices = np.argsort(avg_probs)[::-1]
    for idx in gt_indices:
        rank = np.where(sorted_indices == idx)[0][0] + 1  # 1-indexed
        ranks.append(1.0 / rank)
    mrr = np.mean(ranks)
    
    # 6. Jaccard similarity: intersection / union of top-k and ground truth
    intersection = len(set(top_k_indices) & set(gt_indices))
    union = len(set(top_k_indices) | set(gt_indices))
    jaccard = intersection / union if union > 0 else 0
    
    return {
        'building': building['name'],
        'num_styles': k,
        'ground_truth_styles': building['styles'],
        'ground_truth_indices': gt_indices,
        'top_k_indices': top_k_indices.tolist(),
        'top_k_styles': [CLASSES[i] for i in top_k_indices],
        'top_k_probs': top_k_probs.tolist(),
        'all_probs': avg_probs.tolist(),
        'all_correct': all_correct,
        'any_correct': any_correct,
        'num_correct': num_correct,
        'pct_correct': pct_correct,
        'mean_correct_prob': mean_correct_prob,
        'correct_probs': correct_probs,
        'mrr': mrr,
        'jaccard': jaccard,
    }


def generate_visualizations(results: list, output_dir: Path):
    """Generate visualization plots for hybrid evaluation results."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualizations (matplotlib not available)")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by pct_correct for ranking
    sorted_results = sorted(results, key=lambda x: x['pct_correct'], reverse=True)
    
    # === Figure 1: Building Performance Ranking ===
    fig, ax = plt.subplots(figsize=(14, 10))
    
    buildings = [r['building'] for r in sorted_results]
    pct_correct = [r['pct_correct'] * 100 for r in sorted_results]
    mean_probs = [r['mean_correct_prob'] * 100 for r in sorted_results]
    
    x = np.arange(len(buildings))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, pct_correct, width, label='% Correct Styles in Top-k', color='#2ecc71')
    bars2 = ax.barh(x + width/2, mean_probs, width, label='Mean Prob. of Correct Styles (%)', color='#3498db')
    
    ax.set_xlabel('Score (%)', fontsize=12)
    ax.set_ylabel('Building', fontsize=12)
    ax.set_title('Hybrid Architecture Recognition Performance by Building', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(buildings, fontsize=9)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars1, pct_correct):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'building_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'building_performance.png'}")
    
    # === Figure 2: ALL Buildings - Individual Probability Distributions ===
    num_buildings = len(sorted_results)
    cols = 5
    rows = (num_buildings + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if num_buildings > 1 else [axes]
    
    for idx, result in enumerate(sorted_results):
        ax = axes[idx]
        probs = np.array(result['all_probs'])
        top_indices = np.argsort(probs)[::-1][:6]
        
        styles = [CLASSES[i].replace(' architecture', '').replace(' style', '') for i in top_indices]
        style_probs = probs[top_indices] * 100
        
        colors = ['#2ecc71' if i in result['ground_truth_indices'] else '#3498db' for i in top_indices]
        
        bars = ax.barh(range(len(styles)), style_probs, color=colors)
        ax.set_yticks(range(len(styles)))
        ax.set_yticklabels(styles, fontsize=7)
        ax.set_xlabel('Probability (%)', fontsize=8)
        
        # Mark if all correct
        status = " ✓" if result['all_correct'] else ""
        ax.set_title(f"{result['building']}{status}", fontsize=8, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.tick_params(axis='both', labelsize=7)
    
    # Hide extra subplots
    for idx in range(num_buildings, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    correct_patch = mpatches.Patch(color='#2ecc71', label='Ground Truth Style')
    other_patch = mpatches.Patch(color='#3498db', label='Other Predicted')
    fig.legend(handles=[correct_patch, other_patch], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10)
    
    plt.suptitle('Hybrid Architecture Recognition - All Buildings', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / 'all_buildings_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'all_buildings_predictions.png'}")
    
    # === Figure 3: Best Examples (Top 5) ===
    best_examples = [r for r in sorted_results if r['all_correct']]
    best_examples = sorted(best_examples, key=lambda x: x['mean_correct_prob'], reverse=True)[:5]
    
    if best_examples:
        fig, axes = plt.subplots(1, len(best_examples), figsize=(5*len(best_examples), 6))
        if len(best_examples) == 1:
            axes = [axes]
        
        for ax, result in zip(axes, best_examples):
            probs = np.array(result['all_probs'])
            top_indices = np.argsort(probs)[::-1][:6]
            
            styles = [CLASSES[i].replace(' architecture', '').replace(' style', '') for i in top_indices]
            style_probs = probs[top_indices] * 100
            
            colors = ['#2ecc71' if i in result['ground_truth_indices'] else '#3498db' for i in top_indices]
            
            bars = ax.barh(range(len(styles)), style_probs, color=colors)
            ax.set_yticks(range(len(styles)))
            ax.set_yticklabels(styles, fontsize=10)
            ax.set_xlabel('Probability (%)', fontsize=11)
            ax.set_title(f"{result['building']}\n({result['num_styles']} styles)", fontweight='bold', fontsize=11)
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            
            # Add value labels
            for bar, val in zip(bars, style_probs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=9)
            
            correct_patch = mpatches.Patch(color='#2ecc71', label='Ground Truth')
            other_patch = mpatches.Patch(color='#3498db', label='Other')
            ax.legend(handles=[correct_patch, other_patch], loc='lower right', fontsize=9)
        
        plt.suptitle('Best Hybrid Recognition Examples (Top 5)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'best_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'best_examples.png'}")
    
    # === Figure 4: Metrics Summary ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = {
        'All Correct\n(Top-k)': np.mean([r['all_correct'] for r in results]) * 100,
        'Any Correct\n(Top-k)': np.mean([r['any_correct'] for r in results]) * 100,
        '% Styles\nCorrect': np.mean([r['pct_correct'] for r in results]) * 100,
        'Mean Prob.\nCorrect Styles': np.mean([r['mean_correct_prob'] for r in results]) * 100,
        'Mean Reciprocal\nRank': np.mean([r['mrr'] for r in results]) * 100,
        'Jaccard\nSimilarity': np.mean([r['jaccard'] for r in results]) * 100,
    }
    
    bars = ax.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6', '#8e44ad'])
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Overall Hybrid Recognition Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metrics_summary.png'}")


def print_summary(results: list):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("HYBRID ARCHITECTURE EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal buildings evaluated: {len(results)}")
    print(f"Total images: {sum(r['num_styles'] for r in results) * 2}")  # 2 images per building
    
    # Overall metrics
    all_correct_rate = np.mean([r['all_correct'] for r in results]) * 100
    any_correct_rate = np.mean([r['any_correct'] for r in results]) * 100
    pct_correct_avg = np.mean([r['pct_correct'] for r in results]) * 100
    mean_prob_avg = np.mean([r['mean_correct_prob'] for r in results]) * 100
    mrr_avg = np.mean([r['mrr'] for r in results]) * 100
    jaccard_avg = np.mean([r['jaccard'] for r in results]) * 100
    
    print(f"\n{'Metric':<35} {'Score':>10}")
    print("-" * 50)
    print(f"{'All Correct (all GT in top-k)':<35} {all_correct_rate:>9.1f}%")
    print(f"{'Any Correct (any GT in top-k)':<35} {any_correct_rate:>9.1f}%")
    print(f"{'% Styles Correct (avg)':<35} {pct_correct_avg:>9.1f}%")
    print(f"{'Mean Prob. of Correct Styles':<35} {mean_prob_avg:>9.1f}%")
    print(f"{'Mean Reciprocal Rank':<35} {mrr_avg:>9.1f}%")
    print(f"{'Jaccard Similarity':<35} {jaccard_avg:>9.1f}%")
    
    # Best and worst performers
    sorted_results = sorted(results, key=lambda x: x['pct_correct'], reverse=True)
    
    print(f"\n{'BEST PERFORMERS (100% Correct)':}")
    print("-" * 50)
    best = [r for r in sorted_results if r['all_correct']]
    for r in best[:5]:
        styles_str = ' + '.join([s.replace(' architecture', '') for s in r['ground_truth_styles']])
        print(f"  {r['building']}: {styles_str}")
        probs_str = ', '.join([f"{p*100:.1f}%" for p in r['correct_probs']])
        print(f"    Probabilities: {probs_str}")
    
    print(f"\n{'MOST CHALLENGING (Lowest % Correct)':}")
    print("-" * 50)
    for r in sorted_results[-5:]:
        styles_str = ' + '.join([s.replace(' architecture', '') for s in r['ground_truth_styles']])
        pred_str = ' + '.join([s.replace(' architecture', '') for s in r['top_k_styles']])
        print(f"  {r['building']}: {r['pct_correct']*100:.0f}% correct")
        print(f"    Expected: {styles_str}")
        print(f"    Predicted: {pred_str}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid architecture recognition")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="Path to model weights")
    parser.add_argument("--csv_path", type=str, default="hybrid_buildings/buildings.csv",
                        help="Path to hybrid buildings CSV")
    parser.add_argument("--images_dir", type=str, default="hybrid_buildings",
                        help="Directory containing hybrid building images")
    parser.add_argument("--output_dir", type=str, default="hybrid_results",
                        help="Output directory for results and visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    model = load_model(args.model_path, args.device)
    transform = get_transform()
    
    # Load hybrid buildings
    csv_path = Path(args.csv_path)
    images_dir = Path(args.images_dir)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    buildings = load_hybrid_buildings(csv_path, images_dir)
    print(f"\nLoaded {len(buildings)} hybrid buildings")
    
    # Evaluate each building
    results = []
    for building in buildings:
        result = evaluate_building(building, model, transform, args.device)
        if result:
            results.append(result)
    
    if not results:
        print("No results to report")
        return
    
    # Print summary
    print_summary(results)
    
    # Generate visualizations
    output_dir = Path(args.output_dir)
    generate_visualizations(results, output_dir)
    
    # Save detailed results as JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()

