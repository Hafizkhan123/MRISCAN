import sys
import os

# Parse the filename-based ground truth and the known predictions from earlier test run
# to infer the label ordering

test_data = [
    # (filename, ground_truth_from_prefix, predicted_index_from_earlier_run)
    ('Tr-gl_0017.jpg', 'glioma', 2),
    ('Tr-me_0017.jpg', 'meningioma', 3),
    ('Tr-no_0019.jpg', 'notumor', 1),
    ('Tr-pi_0029.jpg', 'pituitary', 3),  # Wrong prediction, but helps us map
    ('Te-gl_0014.jpg', 'glioma', 2),
    ('Te-me_0010.jpg', 'meningioma', 3),
    ('Te-me_0013.jpg', 'meningioma', 3),
    ('Te-no_0014.jpg', 'notumor', 1),
]

# Build a mapping: index -> label based on what we know
index_to_label = {}
conflicts = []

for fname, ground_truth, predicted_idx in test_data:
    if predicted_idx in index_to_label:
        if index_to_label[predicted_idx] != ground_truth:
            conflicts.append(f"{fname}: index {predicted_idx} mapped to both {index_to_label[predicted_idx]} and {ground_truth}")
        # else: consistent
    else:
        index_to_label[predicted_idx] = ground_truth

print("Inferred label mapping from test predictions:")
print(f"  Index -> Label mapping:")
for idx in sorted(index_to_label.keys()):
    print(f"    {idx} -> {index_to_label[idx]}")

if conflicts:
    print(f"\nConflicts found (label at an index doesn't match multiple images):")
    for c in conflicts:
        print(f"  {c}")
    print(f"\nThis suggests either:")
    print(f"  1. Some model predictions are wrong (overfit/poor model)")
    print(f"  2. The filename prefix doesn't match ground truth")
    print(f"  3. The preprocessing is incorrect")

# Build the full class_labels list
# We know indices 0,1,2,3 map to 4 classes
# From the data: 1->notumor, 2->glioma, 3->meningioma
# So index 0 must be pituitary (the remaining class)

base_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
proposed_order = ['?', '?', '?', '?']

for idx, label in index_to_label.items():
    if idx < 4:
        proposed_order[idx] = label

# Fill in the missing one
remaining = set(base_labels) - set(index_to_label.values())
for i, p in enumerate(proposed_order):
    if p == '?':
        if remaining:
            proposed_order[i] = remaining.pop()

print(f"\nProposed class_labels order (based on inferred mapping):")
print(f"  class_labels = {proposed_order}")

print(f"\nTo fix the issue, update main.py:")
print(f"  Replace: class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']")
print(f"  With:    class_labels = {proposed_order}")
