import os
import io
import base64
import cv2
import numpy as np
import json
from PIL import Image
import time
import logging # Use standard logging

from flask import Flask, request, jsonify
from flask_cors import CORS
# Use headless version of opencv for servers if possible
# Ensure controlnet_aux is installed via requirements.txt
from controlnet_aux import MidasDetector, LineartDetector, OpenposeDetector, HEDdetector

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Loading ---
# Models will be loaded into memory when first requested.
# On Render's free tier, the instance might sleep, causing models to be
# unloaded. They will be re-downloaded/re-loaded on the next request,
# adding to the cold start time.
models_cache = {
    'depth_midas': None,
    'lineart': None,
    'openpose': None,
    'scribble': None, # HEDdetector is used for scribble
    'canny': "opencv" # Canny uses OpenCV directly
}
# Keep track of loaded models to avoid redundant loading messages
models_loaded = set()

# Lock for thread-safe model loading (though less critical with gunicorn workers)
# from threading import Lock
# model_load_lock = Lock()

def get_model(model_type):
    # Use global cache. Consider thread safety if not using separate gunicorn workers.
    global models_cache
    global models_loaded
    # with model_load_lock: # Optional lock
    if models_cache.get(model_type) is None and model_type != 'canny':
        logging.info(f"Loading {model_type} model...")
        try:
            if model_type == 'depth_midas':
                models_cache[model_type] = MidasDetector.from_pretrained("lllyasviel/ControlNet")
            elif model_type == 'lineart':
                models_cache[model_type] = LineartDetector.from_pretrained("lllyasviel/ControlNet")
            elif model_type == 'openpose':
                models_cache[model_type] = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            elif model_type == 'scribble':
                # Use HEDdetector for scribble
                models_cache[model_type] = HEDdetector.from_pretrained("lllyasviel/ControlNet")
            else:
                 logging.warning(f"Attempted to load unknown model type: {model_type}")
                 return None # Unknown model

            logging.info(f"{model_type} model loaded successfully.")
            models_loaded.add(model_type)

        except Exception as e:
            logging.error(f"Error loading model {model_type}: {str(e)}", exc_info=True)
            models_cache[model_type] = None # Ensure it remains None if loading failed
            return None

    elif model_type not in models_loaded and model_type != 'canny':
        # Log if model exists but wasn't logged as loaded in this session (e.g. after cold start)
        logging.info(f"Using pre-loaded {model_type} model.")
        models_loaded.add(model_type)

    return models_cache.get(model_type)

# --- Image Processing Function ---
def process_image(image_data, preprocessor_type, resolution=512):
    try:
        logging.info(f"Processing with {preprocessor_type} at {resolution}px resolution...")
        # Decode base64 image
        if ',' not in image_data:
             raise ValueError("Invalid base64 image data format.")
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure image is RGB

        # Resize image while maintaining aspect ratio
        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError("Invalid image dimensions.")
        ratio = min(resolution / w, resolution / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        # Ensure dimensions are at least 1px
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logging.info(f"Resized image to {new_w}x{new_h}")

        # Get the appropriate model/method
        model_or_method = get_model(preprocessor_type)
        if model_or_method is None:
             raise RuntimeError(f"Model/method for {preprocessor_type} failed to load or is unavailable.")

        # Process based on type
        result = None
        if preprocessor_type == 'canny':
            logging.info("Applying Canny using OpenCV...")
            img_array = np.array(resized_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            result = Image.fromarray(edges)
            logging.info("Canny processing complete.")
        elif preprocessor_type == 'scribble':
             logging.info("Applying Scribble (HED) using model...")
             detector = model_or_method
             result = detector(resized_image, scribble=True) # Use scribble=True for HED based scribble
             logging.info("Scribble processing complete.")
        elif model_or_method != "opencv": # Handle other model-based processors
            logging.info(f"Applying {preprocessor_type} using model...")
            result = model_or_method(resized_image)
            logging.info(f"{preprocessor_type} processing complete.")
        else:
             # Should not happen if model loading is correct, but good to have a fallback
             raise ValueError(f"Unsupported or unknown preprocessor type: {preprocessor_type}")


        if not isinstance(result, Image.Image):
             raise TypeError(f"Processing result for {preprocessor_type} is not a PIL Image.")

        # Convert result to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logging.info(f"Finished encoding result for {preprocessor_type}.")

        return f'data:image/png;base64,{img_str}'
    except Exception as e:
        logging.error(f"Error processing image with {preprocessor_type}: {str(e)}", exc_info=True)
        raise e # Re-raise to be caught by the route handler

# --- Flask App Setup ---
app = Flask(__name__)
# Configure CORS for your frontend URL (replace with your actual frontend if not controlnet.pages.dev)
# Or use '*' for development, but be more specific for production if possible.
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Apply CORS to /api routes

# --- API Routes ---
@app.route('/')
def home():
    # Simple route to check if the server is running
    return "ControlNet Preprocessor Backend is running."

@app.route('/api/process', methods=['POST'])
def process_request():
    logging.info("Received request on /api/process")
    start_time = time.time()
    try:
        data = request.json
        if not data:
             logging.warning("Received empty JSON data.")
             return jsonify({'status': 'error', 'message': 'No JSON data received'}), 400

        # Handle test request (for connection check)
        if data.get('test', False):
            logging.info("Received test request.")
            return jsonify({'status': 'success', 'message': 'Connection test successful'})

        images = data.get('images', [])
        preprocessor_types = data.get('preprocessor_types', [])
        resolution = data.get('resolution', 512)
        logging.info(f"Processing {len(images)} image(s) with preprocessors: {preprocessor_types}, resolution: {resolution}px")

        if not images or not preprocessor_types:
             logging.warning("Missing 'images' or 'preprocessor_types' in request.")
             return jsonify({'status': 'error', 'message': 'Missing images or preprocessor_types'}), 400

        results = {}
        for i, image_data in enumerate(images):
            image_key = f'image_{i}'
            logging.info(f"--- Processing {image_key} ---")
            image_results = {}
            for preprocessor_type in preprocessor_types:
                try:
                    result = process_image(image_data, preprocessor_type, resolution)
                    image_results[preprocessor_type] = result
                except Exception as e:
                    logging.error(f"Failed to process {image_key} with {preprocessor_type}: {str(e)}")
                    image_results[preprocessor_type] = {
                        'error': f"Failed processing {preprocessor_type}: {str(e)}" # Send error details back
                    }
            results[image_key] = image_results
            logging.info(f"--- Finished processing {image_key} ---")

        end_time = time.time()
        logging.info(f"Processing finished successfully in {end_time - start_time:.2f} seconds.")
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error in /api/process endpoint after {end_time - start_time:.2f} seconds: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"Internal server error: {str(e)}"
        }), 500

# Note: Removed the /shutdown route as it's not applicable to Render deployment.

# --- Main Execution ---
# Gunicorn will be used to run the app in production (specified in Procfile or Render start command)
# This block is mainly for local development testing (e.g., python app.py)
if __name__ == '__main__':
    # Use waitress or Flask's dev server for local testing.
    # For production on Render, Gunicorn is recommended.
    port = int(os.environ.get('PORT', 5000)) # Render sets the PORT env var
    logging.info(f"Starting Flask server on port {port}...")
    # Set debug=False for production/testing on Render
    app.run(host='0.0.0.0', port=port, debug=False)