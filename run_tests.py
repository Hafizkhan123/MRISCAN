import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import predict_tumor

# Candidate folders to search for images
candidates = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAMPLRMRIIMAGES')),
              os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads')),
              os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))]

image_paths = []
for c in candidates:
    if os.path.isdir(c):
        for root, _, files in os.walk(c):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, f))
    if len(image_paths) >= 8:
        break

# Keep a small number of images for the quick test
image_paths = image_paths[:8]

print('Found images for test:')
for p in image_paths:
    print(' -', p)

print('\nRunning predictions:')
for p in image_paths:
    try:
        res = predict_tumor(p)
        # predict_tumor returns (label, confidence)
        if isinstance(res, tuple) and len(res) >= 2:
            label, conf = res[0], res[1]
            print(f"{os.path.basename(p)} -> {label}, confidence={conf*100:.2f}%")
        else:
            print(f"{os.path.basename(p)} -> unexpected output: {res}")
    except Exception as e:
        print(f"{os.path.basename(p)} -> prediction error: {e}")
