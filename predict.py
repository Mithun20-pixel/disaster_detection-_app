"""
Standalone prediction script.

Usage:
    python predict.py --image path/to/satellite.png
    python predict.py --image path/to/satellite.png --model saved_models/best_classifier.pth
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CLASS_NAMES
from utils.preprocessing import load_and_preprocess
from utils.prediction import predict_damage_classification, predict_damage_segmentation
from utils.visualization import (
    create_damage_heatmap,
    create_segmentation_overlay,
    draw_damage_bboxes,
    create_severity_map,
)


def main():
    parser = argparse.ArgumentParser(description="Predict disaster damage from a satellite image")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model checkpoint")
    parser.add_argument("--mode", type=str, default="classification",
                        choices=["classification", "segmentation", "both"])
    parser.add_argument("--output", type=str, default="output_result.png", help="Output path")
    args = parser.parse_args()

    print("=" * 60)
    print("  🛰️  Disaster Damage Detection — Inference")
    print("=" * 60)

    # Load image
    image = load_and_preprocess(args.image)
    print(f"📸 Loaded: {args.image}  ({image.shape})")

    if args.mode in ("classification", "both"):
        result = predict_damage_classification(image, model_path=args.model)
        print(f"\n🔍 Classification Result:")
        print(f"   Label:      {result['label']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        for name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            print(f"   {name:>12s}: {bar} {prob:.1%}")

        # Generate heatmap
        heatmap = create_damage_heatmap(image, result["confidence"] if result["class_index"] == 1 else 0.1)
        severity = create_severity_map(result["confidence"] if result["class_index"] == 1 else 0.1)

    if args.mode in ("segmentation", "both"):
        mask, damage_ratio = predict_damage_segmentation(image, model_path=args.model)
        print(f"\n🗺️  Segmentation Result:")
        print(f"   Damage ratio: {damage_ratio:.1%}")

        overlay = create_segmentation_overlay(image, mask)
        bbox_img = draw_damage_bboxes(image, mask)

    # Save result
    if args.mode == "classification":
        cv2.imwrite(args.output, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    elif args.mode == "segmentation":
        cv2.imwrite(args.output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[1].imshow(heatmap)
        axes[1].set_title(f"Heatmap ({result['label']})")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Segmentation ({damage_ratio:.1%} damaged)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n💾 Result saved → {args.output}")


if __name__ == "__main__":
    main()
