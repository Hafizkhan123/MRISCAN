import sys
import os
from collections import defaultdict
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import predict_tumor

# Test directory
test_dir = r'd:\brain_tumor_data\Testing'

if not os.path.isdir(test_dir):
    print(f"ERROR: Test directory not found: {test_dir}")
    sys.exit(1)

# Map directory names to labels (assuming folder structure: Testing/glioma/, Testing/meningioma/, etc.)
label_dirs = {}
for item in os.listdir(test_dir):
    item_path = os.path.join(test_dir, item)
    if os.path.isdir(item_path):
        label_dirs[item.lower()] = item_path

print(f"Found label directories: {list(label_dirs.keys())}\n")

# Collect all test images with ground truth labels
test_images = []
for label_name, label_path in label_dirs.items():
    image_files = []
    for root, _, files in os.walk(label_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, fname)
                image_files.append((full_path, fname, label_name))
    test_images.extend(image_files)
    print(f"Found {len(image_files)} images for label '{label_name}'")

print(f"\nTotal images found: {len(test_images)}\n")

if not test_images:
    print("No images found in test directory!")
    sys.exit(1)

# Run predictions and collect results
results_by_label = defaultdict(list)
overall_correct = 0
overall_total = 0

print("Running predictions...")
print("=" * 80)

for img_path, fname, ground_truth in test_images:
    try:
        predicted_label, confidence = predict_tumor(img_path)
        
        # Extract just the label name (remove "Tumor: " prefix if present)
        if "Tumor: " in predicted_label:
            predicted_label_clean = predicted_label.replace("Tumor: ", "").strip()
        else:
            predicted_label_clean = predicted_label.replace("NO TUMOR", "notumor").strip()
        
        # Check if prediction matches ground truth
        is_correct = predicted_label_clean.lower() == ground_truth.lower()
        
        results_by_label[ground_truth].append({
            'file': fname,
            'predicted': predicted_label_clean,
            'confidence': confidence,
            'correct': is_correct
        })
        
        overall_correct += 1 if is_correct else 0
        overall_total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {ground_truth:15} -> {predicted_label_clean:15} ({confidence*100:5.1f}%) {fname}")
        
    except Exception as e:
        print(f"✗ {ground_truth:15} -> ERROR: {str(e)[:50]} {fname}")
        results_by_label[ground_truth].append({
            'file': fname,
            'predicted': 'ERROR',
            'confidence': 0,
            'correct': False
        })
        overall_total += 1

print("=" * 80)
print()

# Generate summary report
print("SUMMARY REPORT")
print("=" * 80)

for label in sorted(results_by_label.keys()):
    results = results_by_label[label]
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{label.upper()}")
    print(f"  Total: {total} images")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    # Show incorrect predictions
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"  Incorrect predictions ({len(incorrect)}):")
        for r in incorrect[:5]:  # Show first 5 incorrect
            print(f"    - {r['file']}: predicted as {r['predicted']}")
        if len(incorrect) > 5:
            print(f"    ... and {len(incorrect) - 5} more")

print(f"\n{'=' * 80}")
print(f"OVERALL ACCURACY: {overall_correct}/{overall_total} ({overall_correct/overall_total*100:.1f}%)")
print(f"{'=' * 80}")
