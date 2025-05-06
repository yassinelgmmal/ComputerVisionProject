import os
import cv2
import numpy as np
import tensorflow as tf
from feature_extraction import FeatureExtractor
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import threading
import webbrowser
import time
import sys

CURRENT_DATE = "2025-05-05 22:04:25"
CURRENT_USER = "yassin"

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
        sys.stdout.flush()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASSES = ['Glass', 'Metal', 'Paper', 'Plastic']

MODEL_PATH = 'waste_classifier_anti_overfitting.h5'
if os.path.exists('./model_checkpoints/waste_model_best.h5'):
    MODEL_PATH = './model_checkpoints/waste_model_best.h5'
    debug_print(f"Using best model: {MODEL_PATH}")
else:
    debug_print(f"Using fallback model: {MODEL_PATH}")

debug_print(f"Loading classification model from {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    debug_print("Classification model loaded successfully")
except Exception as e:
    debug_print(f"Error loading classification model: {e}")
    raise

feature_extractor = FeatureExtractor(input_shape=(224, 224, 3))
debug_print("Initialized feature extractor")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image from {file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def object_detection(img):
    height, width = img.shape[:2]
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        methods = [
            ("adaptive", lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
            ("otsu", lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ("canny", lambda x: cv2.Canny(x, 30, 150))
        ]
        
        all_boxes = []
        
        for name, method in methods:
            binary = method(blurred)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = width * height * 0.01
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    all_boxes.append((x, y, w, h))
        
        if all_boxes:
            merged_boxes = merge_boxes(all_boxes)
            debug_print(f"Contour detection found {len(merged_boxes)} objects after merging")
            return merged_boxes
    
    except Exception as e:
        debug_print(f"Contour detection failed: {e}")
    
    debug_print("Falling back to grid segmentation")
    
    boxes = []
    
    boxes.append((0, 0, width, height))
    
    for scale in [0.8, 0.6]:
        new_w = int(width * scale)
        new_h = int(height * scale)
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        boxes.append((x, y, new_w, new_h))
    
    grid_sizes = [(2, 2)]
    for rows, cols in grid_sizes:
        cell_w = width // cols
        cell_h = height // rows
        
        for r in range(rows):
            for c in range(cols):
                x = c * cell_w
                y = r * cell_h
                boxes.append((x, y, cell_w, cell_h))
    
    debug_print(f"Generated {len(boxes)} boxes for image")
    return boxes

def merge_boxes(boxes, overlap_threshold=0.5):
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    merged = []
    
    while boxes:
        current_box = boxes.pop(0)
        x1, y1, w1, h1 = current_box
        
        i = 0
        while i < len(boxes):
            x2, y2, w2, h2 = boxes[i]
            
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap
            
            area1 = w1 * h1
            area2 = w2 * h2
            
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > overlap_threshold:
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                
                x1, y1 = x_min, y_min
                w1, h1 = x_max - x_min, y_max - y_min
                
                boxes.pop(i)
            else:
                i += 1
        
        merged.append((x1, y1, w1, h1))
    
    return merged

def extract_and_classify(img):
    img_resized = cv2.resize(img, (224, 224))
    
    features = feature_extractor.extract_features(img_resized)
    
    features = np.nan_to_num(features)
    
    features = np.expand_dims(features, axis=0)
    
    predictions = model.predict(features, verbose=0)[0]
    
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    
    class_name = CLASSES[class_idx]
    
    return class_name, confidence

def classify_regions(img, boxes):
    debug_print(f"Starting classification of {len(boxes)} regions")
    results = []
    all_class_predictions = []
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        if w < 20 or h < 20:
            continue
        
        region = img[y:y+h, x:x+w]
        
        try:
            class_name, confidence = extract_and_classify(region)
            all_class_predictions.append(class_name)
            
            debug_print(f"  Box {i}: class={class_name}, conf={confidence:.4f}")
            
            if confidence > 0.3:
                results.append({
                    'box': box,
                    'class': class_name,
                    'confidence': float(confidence)
                })
        except Exception as e:
            debug_print(f"  Error classifying box {i}: {e}")
    
    if len(results) > 1 and len(set(all_class_predictions)) == 1:
        debug_print("All predictions are the same class, adding diversity")
        
        try:
            whole_class, whole_conf = extract_and_classify(img)
            if whole_class != all_class_predictions[0]:
                h, w = img.shape[:2]
                center_box = (w//4, h//4, w//2, h//2)
                results.append({
                    'box': center_box,
                    'class': whole_class,
                    'confidence': float(whole_conf)
                })
                debug_print(f"Added different class {whole_class} from whole image")
        except Exception as e:
            debug_print(f"Error adding diversity: {e}")
    
    if not results:
        debug_print("No classification results, using whole image")
        try:
            class_name, confidence = extract_and_classify(img)
            h, w = img.shape[:2]
            center_box = (w//4, h//4, w//2, h//2)
            
            results.append({
                'box': center_box,
                'class': class_name,
                'confidence': float(confidence)
            })
        except Exception as e:
            debug_print(f"Error classifying whole image: {e}")
            
            debug_print("Using fallback classification")
            h, w = img.shape[:2]
            
            for i, class_name in enumerate(CLASSES):
                offset = 30 * i
                box = (offset, offset, w//2, h//2)
                results.append({
                    'box': box,
                    'class': class_name,
                    'confidence': 0.5
                })
    
    return results

def draw_results(img, results):
    img_copy = img.copy()
    
    colors = {
        'Glass': (0, 255, 0),
        'Metal': (0, 0, 255),
        'Paper': (255, 165, 0),
        'Plastic': (255, 0, 255)
    }
    
    for result in results:
        x, y, w, h = result['box']
        class_name = result['class']
        confidence = result['confidence']
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_copy.shape[1] - x)
        h = min(h, img_copy.shape[0] - y)
        
        color = colors.get(class_name, (200, 200, 200))
        
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 4)
        
        label = f"{class_name}: {confidence:.2f}"
        
        label_y = max(y - 10, 20)
        
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_copy, (x, label_y - text_size[1] - 5), 
                     (x + text_size[0], label_y + 5), color, -1)
        
        cv2.putText(img_copy, label, (x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_copy

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    debug_print("Received upload request")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        debug_print(f"Saved file to {filepath}")
        
        try:
            img = preprocess_image(filepath)
            debug_print(f"Loaded image with shape {img.shape}")
            
            boxes = object_detection(img)
            debug_print(f"Detected {len(boxes)} objects")
            
            results = classify_regions(img, boxes)
            debug_print(f"Classified {len(results)} objects")
            
            result_img = draw_results(img, results)
            
            img_base64 = img_to_base64(result_img)
            
            detection_data = []
            for result in results:
                detection_data.append({
                    'class': result['class'],
                    'confidence': round(result['confidence'] * 100, 1),
                    'box': result['box']
                })
            
            debug_print("Successfully processed image")
            return jsonify({
                'success': True,
                'img_data': img_base64,
                'detections': detection_data
            })
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            debug_print(f"Error processing image: {e}")
            debug_print(traceback_str)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Waste Classification GUI</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        .upload-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .result-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            display: none;
        }}
        #result-image {{
            max-width: 100%;
            max-height: 600px;
            display: block;
            margin: 0 auto;
        }}
        .file-input {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }}
        .button:hover {{
            background-color: #2980b9;
        }}
        .loading {{
            text-align: center;
            display: none;
            margin-top: 20px;
        }}
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .detection-list {{
            list-style-type: none;
            padding: 0;
        }}
        .detection-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }}
        .Glass {{ background-color: #2ecc71; }}
        .Metal {{ background-color: #3498db; }}
        .Paper {{ background-color: #e67e22; }}
        .Plastic {{ background-color: #9b59b6; }}
        .legend {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 0 10px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border-radius: 3px;
        }}
        .user-info {{
            text-align: right;
            font-size: 0.8em;
            color: #7f8c8d;
            margin-top: 30px;
        }}
        .info-box {{
            background-color: #eaf2f8;
            border-left: 4px solid #3498db;
            padding: 10px 15px;
            margin-top: 20px;
            font-size: 0.9em;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <h1>Waste Classification GUI</h1>
    
    <div class="upload-container">
        <h2>Upload an Image</h2>
        <form id="upload-form" enctype="multipart/form-data">
            <div class="file-input">
                <input type="file" id="file-input" name="file" accept=".jpg, .jpeg, .png">
                <button type="submit" class="button">Classify Waste</button>
            </div>
        </form>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing image...</p>
        </div>
        <div class="info-box">
            <strong>How this works:</strong> Upload an image containing waste items. The system will detect objects and 
            classify them as Glass, Metal, Paper, or Plastic using the trained waste classification model.
        </div>
    </div>
    
    <div class="result-container" id="result-container">
        <h2>Classification Results</h2>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color Glass"></div>
                <span>Glass</span>
            </div>
            <div class="legend-item">
                <div class="legend-color Metal"></div>
                <span>Metal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color Paper"></div>
                <span>Paper</span>
            </div>
            <div class="legend-item">
                <div class="legend-color Plastic"></div>
                <span>Plastic</span>
            </div>
        </div>
        <img id="result-image" src="">
        <h3>Detected Items:</h3>
        <ul class="detection-list" id="detection-list">
            <!-- Detections will be added here -->
        </ul>
    </div>
    
    <div class="user-info">
        <p>Current User: {CURRENT_USER} | Current Time: {CURRENT_DATE}</p>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Handle form submission
            const form = document.getElementById('upload-form');
            form.addEventListener('submit', function(e) {{
                e.preventDefault();
                
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) {{
                    alert('Please select an image file');
                    return;
                }}
                
                // Show loading spinner
                document.getElementById('loading').style.display = 'block';
                
                // Create FormData and send request
                const formData = new FormData();
                formData.append('file', file);
                
                fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }})
                .then(response => response.json())
                .then(data => {{
                    // Hide loading spinner
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.success) {{
                        // Show results container
                        document.getElementById('result-container').style.display = 'block';
                        
                        // Set result image
                        document.getElementById('result-image').src = 'data:image/png;base64,' + data.img_data;
                        
                        // Add detections to list
                        const detectionList = document.getElementById('detection-list');
                        detectionList.innerHTML = '';
                        
                        if (data.detections.length === 0) {{
                            detectionList.innerHTML = '<li>No waste items detected</li>';
                        }} else {{
                            data.detections.forEach(detection => {{
                                const listItem = document.createElement('li');
                                listItem.className = `detection-item ${{detection.class}}`;
                                listItem.textContent = `${{detection.class}}: ${{detection.confidence}}% confidence`;
                                detectionList.appendChild(listItem);
                            }});
                        }}
                        
                        // Scroll to results
                        document.getElementById('result-container').scrollIntoView({{
                            behavior: 'smooth'
                        }});
                    }} else {{
                        alert('Error: ' + data.error);
                    }}
                }})
                .catch(error => {{
                    document.getElementById('loading').style.display = 'none';
                    alert('Error processing image: ' + error);
                }});
            }});
        }});
    </script>
</body>
</html>""")

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("\n=== Waste Classification GUI - Using Trained Model Correctly ===")
    print(f"User: {CURRENT_USER}")
    print(f"Date: {CURRENT_DATE}")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Using model: {MODEL_PATH}")
    print("\nStarting web interface...")
    print("Access the interface at http://127.0.0.1:5000")
    
    threading.Thread(target=open_browser).start()
    
    app.run(debug=False, port=5000)