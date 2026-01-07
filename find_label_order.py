import sys
import os
from itertools import permutations
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# We'll manually call the prediction function with different label orders
# First, import the model and prediction logic
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

# Load model
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(project_root, 'models', 'model.keras')

print("Loading model...")
from keras.src.layers import Flatten as KerasFlatten
original_compute_output_spec = KerasFlatten.compute_output_spec
def patched_compute_output_spec(self, inputs):
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    return original_compute_output_spec(self, inputs)
KerasFlatten.compute_output_spec = patched_compute_output_spec

model = load_model(model_path, compile=False)
print("Model loaded.\n")

# Collect test images with ground truth from filename
base_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
test_dir = os.path.join(project_root, 'SAMPLRMRIIMAGES')

# Map filename prefixes to ground truth labels
prefix_to_label = {
    'Tr-pi': 'pituitary',
    'Tr-gl': 'glioma',
    'Tr-no': 'notumor',
    'Tr-me': 'meningioma',
    'Te-pi': 'pituitary',
    'Te-gl': 'glioma',
    'Te-no': 'notumor',
    'Te-me': 'meningioma',
}

test_images = []
if os.path.isdir(test_dir):
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract prefix
            prefix = fname[:5]
            if prefix in prefix_to_label:
                full_path = os.path.join(test_dir, fname)
                ground_truth = prefix_to_label[prefix]
                test_images.append((full_path, fname, ground_truth))

# Also check uploads folder
upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
if os.path.isdir(upload_dir):
    for fname in os.listdir(upload_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            prefix = fname[:5]
            if prefix in prefix_to_label:
                full_path = os.path.join(upload_dir, fname)
                ground_truth = prefix_to_label[prefix]
                test_images.append((full_path, fname, ground_truth))

print(f"Found {len(test_images)} test images with ground truth labels.\n")

# Function to predict with a given class label order
def predict_with_labels(image_path, class_labels):
    IMAGE_SIZE = 256
    try:
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img).astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = model.predict(img_array, verbose=0)
        
        # Check if already probabilities
        sums = np.sum(prediction, axis=1)
        is_prob = np.allclose(sums, 1.0, rtol=1e-3, atol=1e-3)
        
        if not is_prob:
            probs = tf.nn.softmax(prediction, axis=1).numpy()
        else:
            probs = prediction
        
        predicted_idx = int(np.argmax(probs, axis=1)[0])
        predicted_label = class_labels[predicted_idx]
        confidence = float(probs[0, predicted_idx])
        
        return predicted_label, confidence, predicted_idx
    except Exception as e:
        print(f"  Error predicting {os.path.basename(image_path)}: {e}")
        return None, None, None

# Test all permutations
print("Testing all permutations of class labels...\n")
best_score = 0
best_labels = None
best_results = None

for perm in permutations(base_labels):
    class_labels = list(perm)
    
    # Test on all images
    correct = 0
    results = []
    
    for img_path, fname, ground_truth in test_images:
        predicted_label, conf, idx = predict_with_labels(img_path, class_labels)
        if predicted_label:
            is_correct = predicted_label == ground_truth
            correct += is_correct
            results.append({
                'file': fname,
                'ground_truth': ground_truth,
                'predicted': predicted_label,
                'idx': idx,
                'correct': is_correct
            })
    
    accuracy = correct / len(test_images) if test_images else 0
    
    if accuracy > best_score:
        best_score = accuracy
        best_labels = class_labels
        best_results = results
    
    # Print permutation info
    if accuracy > 0.5:  # Only print good matches
        print(f"Permutation: {class_labels} -> Accuracy: {accuracy*100:.1f}%")

print(f"\n{'='*70}")
print(f"BEST MATCH FOUND")
print(f"{'='*70}")
print(f"Class labels order: {best_labels}")
print(f"Accuracy on test set: {best_score*100:.1f}%\n")

print("Detailed results with BEST labels:")
print(f"{'File':<25} {'Ground Truth':<15} {'Predicted':<15} {'Index':<5} {'Correct':<8}")
print("-" * 70)
for r in sorted(best_results, key=lambda x: x['file']):
    status = "✓" if r['correct'] else "✗"
    print(f"{r['file']:<25} {r['ground_truth']:<15} {r['predicted']:<15} {r['idx']:<5} {status:<8}")

print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")
print(f"Update class_labels in main.py to:")
print(f"class_labels = {best_labels}")
print(f"{'='*70}")
