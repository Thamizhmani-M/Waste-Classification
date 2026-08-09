from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import os
import cv2
import base64
import numpy as np
import uuid
import datetime
import random 

app = Flask(__name__)
CORS(app) 

# --- CONFIGURATION ---
model = YOLO('yolov8x.pt') 

UPLOAD_FOLDER = 'static/uploads'
CROPS_FOLDER = 'static/crops'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

# --- MUNICIPALITY CATEGORY DATABASE ---
CATEGORY_INFO = {
    'Wet Waste (Organic & Food)': {
        'type': 'Biodegradable', 'time': '2-4 Weeks', 'icon': 'leaf', 'colorClass': 'green',
        'impactScore': 2, 'impactLabel': 'Methane (if buried)', 'binName': 'Green Bin', 'binStyle': 'bin-green',
        'recyclable': True, 'suggestions': [
            'Composting: Can be converted into organic fertilizer to be used in gardens and agriculture.',
            'Biogas: Can be processed to produce cooking gas and generate electricity.'
        ]
    },
    'Wet Waste (Paper)': {
        'type': 'Biodegradable / Recyclable', 'time': '2-6 Weeks', 'icon': 'file-text', 'colorClass': 'green',
        'impactScore': 3, 'impactLabel': 'Deforestation (if wasted)', 'binName': 'Green/Blue Bin', 'binStyle': 'bin-green',
        'recyclable': True, 'suggestions': [
            'Paper Recycling: Old papers can be pulped to create new notebook paper, cardboard boxes, and egg trays.',
            'Composting: Uncoated paper can be added to compost bins.'
        ]
    },
    'Wet Waste (Hood)': {
        'type': 'Biodegradable / Wood', 'time': '1-3 Years', 'icon': 'box', 'colorClass': 'green',
        'impactScore': 3, 'impactLabel': 'Methane (if buried)', 'binName': 'Green Bin', 'binStyle': 'bin-green',
        'recyclable': True, 'suggestions': [
            'Wood Recycling: Can be chipped into biomass fuel pellets or used in particleboards.'
        ]
    },
    'Dry Waste (Plastic waste)': {
        'type': 'Mostly Non-Biodegradable', 'time': '400+ Years', 'icon': 'package', 'colorClass': 'blue',
        'impactScore': 7, 'impactLabel': 'Microplastics', 'binName': 'Blue Bin', 'binStyle': 'bin-blue',
        'recyclable': True, 'suggestions': [
            'Plastics: Can be melted to manufacture plastic chairs, tables, buckets, cans, and used as polymer-modified bitumen in road construction.'
        ]
    },
    'Dry Waste (Bottles/cans)': {
        'type': 'Non-Biodegradable', 'time': '450 Years', 'icon': 'droplet', 'colorClass': 'blue',
        'impactScore': 6, 'impactLabel': 'Microplastics & Ocean Pollution', 'binName': 'Blue Bin', 'binStyle': 'bin-blue',
        'recyclable': True, 'suggestions': [
            'Bottle Recycling: Can be recycled into new bottles, polyester clothing (fleece), and synthetic fibers.'
        ]
    },
    'Dry Waste (CAN)': {
        'type': 'Non-Biodegradable', 'time': '50-200 Years', 'icon': 'trash-2', 'colorClass': 'blue',
        'impactScore': 4, 'impactLabel': 'Landfill Space', 'binName': 'Blue Bin', 'binStyle': 'bin-blue',
        'recyclable': True, 'suggestions': [
            'Can Recycling: Metals and cans can be recycled into new products.',
            'Glass/Ceramics: Repurposed as sand substitute in construction.'
        ]
    },
    'E-Waste': {
        'type': 'Hazardous / Non-Degradable', 'time': '1M+ Years', 'icon': 'battery', 'colorClass': 'orange',
        'impactScore': 10, 'impactLabel': 'Soil Toxicity & Heavy Metals', 'binName': 'Orange Bin', 'binStyle': 'bin-orange',
        'recyclable': True, 'suggestions': [
            'Metals: Precious metals like Gold and Copper can be safely extracted from motherboards and reused in new electronics.',
            'Casings: Plastic computer cases can be melted down and remolded into new appliance bodies.'
        ]
    },
    'Hazardous Waste': {
        'type': 'Dangerous / Sharps', 'time': 'Varies', 'icon': 'alert-triangle', 'colorClass': 'red',
        'impactScore': 8, 'impactLabel': 'Physical Hazard & Infection', 'binName': 'Red Bin', 'binStyle': 'bin-red',
        'recyclable': False, 'suggestions': [
            'For safety reasons, this cannot be recycled into direct consumer products. It must be safely incinerated and disposed of in secure, scientific landfills.'
        ]
    }
}

# --- COCO MAPPINGS ---
COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
    39: 'Bottle', 40: 'CAN', 41: 'Bottle', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'CAN', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
    60: 'Hood', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    70: 'toaster', 71: 'CAN', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'CAN', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def get_waste_mapping(cls_id):
    if cls_id in [73, 59]: return 'Wet Waste (Paper)'
    elif cls_id in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 42, 43, 44, 56, 76, 77, 79]: return 'Dry Waste (Plastic waste)'
    elif cls_id in [39, 41]: return 'Dry Waste (Bottles/cans)'
    elif cls_id in [40, 45, 71, 75]: return 'Dry Waste (CAN)' 
    elif cls_id == 60: return 'Wet Waste (Hood)'
    elif cls_id in [13, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58]: return 'Wet Waste (Organic & Food)'
    elif cls_id in [9, 12, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 74, 78]: return 'E-Waste'
    else: return None

def get_co2_saved(cls_id):
    if cls_id in [39, 41, 24, 25]: return 0.08
    elif cls_id in [40, 45, 75]: return 0.15
    elif cls_id in [73, 56, 57, 59, 60]: return 0.50 if cls_id == 73 else 2.50
    elif cls_id in [42, 44, 76]: return 0.05
    elif cls_id in [64, 65, 66, 67, 74, 78]: return 1.20
    elif cls_id in [62, 63, 68, 69, 70, 72]: return 5.50 if cls_id == 63 else 15.00
    elif cls_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58]: return 0.05
    return 0.02

DASHBOARD_DATA = { 'image': [], 'video': [], 'live': [] }

def get_type_color(colorClass):
    mapping = {'green': 'emerald', 'blue': 'cyan', 'orange': 'orange', 'red': 'red'}
    return mapping.get(colorClass, 'slate')

def save_to_dashboard(source, category_name, exact_coco_item, info, cls_id):
    if source not in DASHBOARD_DATA: return
    co2_val = get_co2_saved(cls_id) 
    
    # Matching Image Logic for Dashboard display
    type_with_bracket = f"{info['type']} ({exact_coco_item})"
    
    event = {
        "name": category_name, # Storing broad category as name to match your request
        "type": type_with_bracket, 
        "time": info['time'],
        "impactScore": info['impactScore'], 
        "typeColor": get_type_color(info['colorClass']),
        "binName": info['binName'], 
        "binStyle": info['binStyle'],
        "timestamp": datetime.datetime.now().isoformat(), 
        "count": 1, 
        "co2Saved": co2_val
    }
    DASHBOARD_DATA[source].append(event)

# --- ROUTES ---

@app.route('/api/detect', methods=['POST'])
def api_detect():
    try:
        data = request.json
        if not data or 'image' not in data: return jsonify({"error": "No image data provided"}), 400

        source_mode = data.get('source', 'image') 
        header, encoded = data['image'].split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: return jsonify({"error": "Decoding failed"}), 400

        h, w, _ = img.shape
        results = model.predict(source=img, conf=0.45, verbose=False) 
        json_results = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # --- CATEGORY MAPPING LOGIC ---
                category_name = get_waste_mapping(cls_id)
                if category_name is None: continue 

                # We still keep the exact COCO item for dashboard brackets, but NOT for the main display
                coco_item = COCO_NAMES.get(cls_id, "Item").title()
                if coco_item.upper() == 'CAN': coco_item = 'CAN'

                info = CATEGORY_INFO[category_name]
                
                # Save to dashboard using broad category name
                save_to_dashboard(source_mode, category_name, coco_item, info, cls_id)

                # Format like Image Logic: Category Name is the primary label
                type_with_bracket = f"{info['type']} ({coco_item})"
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_data = {"top": (y1/h)*100, "left": (x1/w)*100, "w": ((x2-x1)/w)*100, "h": ((y2-y1)/h)*100}

                # Save Crop
                crop_id = str(uuid.uuid4())[:8]
                crop_name = f"crop_{cls_id}_{crop_id}.jpg"
                crop_path = os.path.join(CROPS_FOLDER, crop_name)
                crop_img = img[int(y1):int(y2), int(x1):int(x2)]
                if crop_img.size > 0: cv2.imwrite(crop_path, crop_img)

                json_results.append({
                    "name": category_name, # Changed from coco_item to category_name
                    "category": category_name, 
                    "conf": f"{conf:.1%}", 
                    "box": box_data,
                    "imgCrop": f"/static/crops/{crop_name}", 
                    "type": type_with_bracket, 
                    "time": info['time'], 
                    "icon": info['icon'], 
                    "colorClass": info['colorClass'],
                    "recyclable": info.get('recyclable', True), 
                    "suggestions": info.get('suggestions', []), 
                    "impactScore": info['impactScore'], 
                    "impactLabel": info['impactLabel'], 
                    "binName": info['binName'], 
                    "binStyle": info['binStyle'], 
                    "exact_class": cls_id
                })

        return jsonify(json_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ALL OTHER ROUTES PRESERVED ---

@app.route('/')
def login(): return render_template('login.html')

@app.route('/index')
def index(): return render_template('index.html')

@app.route('/how-it-works')
def how_it_works(): return render_template('how-it-works.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/detect')
def analyse(): return render_template('detect.html')

@app.route('/image-detection')
def image_detection(): return render_template('image_detection.html')

@app.route('/image-result')
def image_result(): return render_template('image_result.html')

@app.route('/video-detection')
def video_detection(): return render_template('video-detection.html')

@app.route('/video-result')
def video_result(): return render_template('video-result.html')

@app.route('/live-detection')
def live_detection(): return render_template('live-detection.html')

@app.route('/geo-analytics')
def geo_analytics(): return render_template('geo_analytics.html')

@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data(): return jsonify(DASHBOARD_DATA)

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({"error": "No video uploaded"}), 400
    file = request.files['video']
    filename = str(uuid.uuid4()) + ".mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({"filename": filename})

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_in_video = set() 
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        results = model.predict(source=frame, conf=0.45, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                category_name = get_waste_mapping(cls_id)
                if not category_name: continue
                
                coco_item = COCO_NAMES.get(cls_id, "Item").title()
                info = CATEGORY_INFO[category_name]
                if coco_item not in detected_in_video:
                    detected_in_video.add(coco_item)
                    save_to_dashboard('video', category_name, coco_item, info, cls_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw bounding box labels using CATEGORY instead of coco_name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_frames(os.path.join(UPLOAD_FOLDER, filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

RECYCLING_EXPLORER_DB = {
    'Wet Waste': [{'item': 'Food Scraps', 'icon': 'apple', 'products': ['Compost', 'Biogas']}],
    'Dry Waste': [{'item': 'Plastics', 'icon': 'package', 'products': ['Road construction', 'Furniture']}],
    'E-Waste': [{'item': 'Circuit Boards', 'icon': 'cpu', 'products': ['Precious metals']}],
    'Hazardous Waste': [{'item': 'Toxic Items', 'icon': 'alert-triangle', 'products': ['Incineration']}]
}

@app.route('/api/recycling_explorer', methods=['GET'])
def get_recycling_explorer_data(): return jsonify(RECYCLING_EXPLORER_DB)

if __name__ == '__main__':
    app.run(debug=True, port=5000)