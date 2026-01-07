from flask import Flask, render_template, request, send_from_directory, url_for
import os
import sys
import warnings

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as ie:
    print(f"Error importing TensorFlow/Keras: {ie}")
    print("Please ensure TensorFlow is installed: pip install tensorflow")
    sys.exit(1)

import numpy as np


#create app
 # Fix: explicit template folder path so Flask can find index.html
 # The project uses a folder called 'TEMPLATES' at the repo root; Flask by
 # default looks for 'templates' relative to app root. Set template_folder
 # so app finds templates regardless of where main.py is executed from.
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TEMPLATES'))
app = Flask(__name__, template_folder=template_dir)

# Debug helper: show the configured templates path so it's easy to confirm at runtime
print(f"Using template folder: {template_dir}")
if not os.path.isdir(template_dir):
    print("Warning: template folder not found. Ensure the 'TEMPLATES' folder exists in the project root or use the default 'templates' name.")

# Load the trained model using an explicit absolute path relative to repo root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(project_root, 'models', 'model.keras')
print(f"\n{'='*60}")
print(f"Project root: {project_root}")
print(f"Looking for model at: {model_path}")
print(f"Model file exists: {os.path.isfile(model_path)}")

model = None
if not os.path.isfile(model_path):
    print(f"ERROR: Model file not found at {model_path}")
else:
    try:
        print("Loading model (this may take 30-60 seconds on first run)...")
        # Patch Keras layers for compatibility with older model format
        # This fixes the Flatten layer issue in TensorFlow 2.20
        from keras.src.layers import Flatten as KerasFlatten
        from keras.src.ops import reshape
        
        original_compute_output_spec = KerasFlatten.compute_output_spec
        def patched_compute_output_spec(self, inputs):
            # Handle case where inputs might be passed as a list
            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]
            return original_compute_output_spec(self, inputs)
        
        KerasFlatten.compute_output_spec = patched_compute_output_spec
        
        model = load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model)}")
        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"ERROR loading model: {type(e).__name__}: {e}")
        print(f"\nNote: Model was saved with an older TensorFlow/Keras version.")
        print(f"Current: TensorFlow {tf.__version__}")
        import traceback
        traceback.print_exc()
        model = None

print(f"{'='*60}\n")

#class labels (corrected order based on model training)
class_labels = ['pituitary', 'notumor', 'glioma', 'meningioma']

#define the uploads folder (use project-level uploads directory so files
# are accessible from the web UI and from other scripts)
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 256
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img).astype('float32')

    # Use VGG16 preprocessing because the model was built on VGG16 base
    from tensorflow.keras.applications.vgg16 import preprocess_input

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    if model is None:
        raise RuntimeError("Model not loaded. Prediction not possible. Check server logs for model load failure.")

    # Get raw predictions from the model
    prediction = model.predict(img_array)

    # If the model does not output probabilities (softmax), convert logits to
    # probabilities so confidence is meaningful.
    try:
        sums = np.sum(prediction, axis=1)
        is_prob = np.allclose(sums, 1.0, rtol=1e-3, atol=1e-3)
    except Exception:
        is_prob = False

    if not is_prob:
        import tensorflow as tf
        probs = tf.nn.softmax(prediction, axis=1).numpy()
    else:
        probs = prediction

    # Log raw predictions for debugging label-order issues
    print(f"Raw model output: {prediction}")
    print(f"Probabilities: {probs}")

    # Primary prediction
    predicted_class_index = int(np.argmax(probs, axis=1)[0])
    confidence_score = float(probs[0, predicted_class_index])

    label = class_labels[predicted_class_index]
    # Log mapping so we can verify label order vs training
    print(f"Predicted index: {predicted_class_index} -> label: {label} (confidence={confidence_score:.4f})")
    display_label = "NO TUMOR" if label == 'notumor' else f"Tumor: {label}"
    return display_label, confidence_score
    
#Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # HANDLE FILE UPLOAD
        file = request.files['file']
        
        if file:
            #save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)
            
            #predict results
            try:
                result, confidence = predict_tumor(file_location)
            except Exception as e:
                # Return useful message to template on failure
                print(f"Prediction failed: {e}")
                return render_template('index.html', result=f"Prediction error: {e}", confidence=None)
            # Build a URL for the uploaded file using url_for so paths are correct
            file_url = url_for('get_uploaded_file', filename=file.filename)
            return render_template(
                'index.html',
                result=result,
                file_path=file_url
            )
    return render_template('index.html', result=None)


#Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#python main
if __name__ == '__main__':
    app.run(debug=True)