"""
Kinemation - Flask Application
================================
Unified Flask app that serves the frontend UI and provides API endpoints
for the 3D pose estimation backend pipeline.

Usage:
    python app.py

The app will start at http://localhost:8080

API Endpoints:
    POST /api/upload     - Upload a video file for processing
    POST /api/webcam     - Upload a webcam recording for processing
    GET  /api/status/<id> - Check processing job status
    GET  /api/download/<f> - Download processed output file
"""

import os
import sys
import uuid
import threading
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory

# --- Path setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(APP_DIR, 'outputs')
BACKEND_DIR = os.path.join(APP_DIR, 'backend')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Add backend to sys.path so we can import from it
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# --- Flask app ---
app = Flask(__name__,
            template_folder=os.path.join(APP_DIR, 'frontend-resources', 'templates'),
            static_folder=os.path.join(APP_DIR, 'frontend-resources', 'assets'),
            static_url_path='/assets')

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

# --- Job tracking ---
# In-memory job store   {job_id: {status, progress, message, phase, output_filename, error}}
jobs = {}


def update_job(job_id, **kwargs):
    """Thread-safe job status update."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)


def run_pipeline(job_id, input_path, output_path):
    """Run the 3D pose estimation pipeline in a background thread."""
    try:
        update_job(job_id, status='processing', progress=0.0,
                   message='Loading models...', phase='Phase 1/4: Setup')

        # Import backend pipeline (deferred to avoid slow startup)
        from main import PoseEstimationPipeline

        pipeline = PoseEstimationPipeline(
            models_dir=os.path.join(BACKEND_DIR, 'models'),
            device='cpu'
        )

        update_job(job_id, progress=0.05,
                   message='Models loaded. Starting processing...',
                   phase='Phase 1/4: Extracting 2D poses')

        # Progress callback for real-time status updates
        def on_progress(progress, message, phase):
            update_job(job_id, progress=progress, message=message, phase=phase)

        # Run the full processing pipeline
        pipeline.process_video(
            video_path=input_path,
            output_path=output_path,
            smoothing_sigma=2.0,
            render_mode='side_by_side',
            export_npy=False,
            show_progress=False,
            progress_callback=on_progress
        )

        pipeline.close()

        output_filename = os.path.basename(output_path)
        update_job(job_id, status='completed', progress=1.0,
                   message='Processing complete!',
                   output_filename=output_filename)

    except Exception as e:
        update_job(job_id, status='failed', error=str(e))


# ==========================
# Page Routes
# ==========================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')


# ==========================
# API Routes
# ==========================

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Accept an uploaded video file and start processing."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    # Validate extension
    allowed_ext = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({'success': False,
                        'error': f'Unsupported format: {ext}. Use MP4, AVI, MOV, MKV, or WEBM.'}), 400

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    safe_name = f"{job_id}_{video.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, safe_name)
    video.save(input_path)

    # Determine output path
    output_name = f"{job_id}_output.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # Initialize job and start processing thread
    jobs[job_id] = {
        'status': 'processing',
        'progress': 0.0,
        'message': 'Video uploaded. Starting pipeline...',
        'phase': 'Initializing',
        'output_filename': None,
        'error': None
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, input_path, output_path), daemon=True)
    thread.start()

    return jsonify({'success': True, 'job_id': job_id})


@app.route('/api/webcam', methods=['POST'])
def api_webcam():
    """Accept a webcam recording blob and start processing."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No recording provided'}), 400

    video = request.files['video']

    # Save recording
    job_id = str(uuid.uuid4())[:8]
    # Webcam recordings come as webm from the browser
    ext = os.path.splitext(video.filename)[1].lower() if video.filename else '.webm'
    if not ext:
        ext = '.webm'
    safe_name = f"{job_id}_webcam{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, safe_name)
    video.save(input_path)

    # Convert webm to mp4 if needed (OpenCV can't always read webm)
    actual_input = input_path
    if ext == '.webm':
        mp4_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_webcam.mp4")
        converted = convert_webm_to_mp4(input_path, mp4_path)
        if converted:
            actual_input = mp4_path

    # Determine output path
    output_name = f"{job_id}_webcam_output.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # Initialize job
    jobs[job_id] = {
        'status': 'processing',
        'progress': 0.0,
        'message': 'Recording uploaded. Starting pipeline...',
        'phase': 'Initializing',
        'output_filename': None,
        'error': None
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, actual_input, output_path), daemon=True)
    thread.start()

    return jsonify({'success': True, 'job_id': job_id})


@app.route('/api/status/<job_id>')
def api_status(job_id):
    """Check the status of a processing job."""
    if job_id not in jobs:
        return jsonify({'status': 'not_found', 'error': 'Job not found'}), 404

    job = jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job.get('progress', 0),
        'message': job.get('message', ''),
        'phase': job.get('phase', ''),
        'output_filename': job.get('output_filename'),
        'error': job.get('error')
    })


@app.route('/api/download/<filename>')
def api_download(filename):
    """Download a processed output file."""
    # Security: only serve from the outputs directory
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_FOLDER, safe_filename)

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(OUTPUT_FOLDER, safe_filename, as_attachment=True)


# ==========================
# Utility Functions
# ==========================

def convert_webm_to_mp4(webm_path, mp4_path):
    """Convert webm to mp4 using OpenCV (frame-by-frame copy).
    Returns True if conversion succeeded, False otherwise."""
    try:
        import cv2
        cap = cv2.VideoCapture(webm_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            cap.release()
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        return os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0

    except Exception:
        return False


# ==========================
# Main
# ==========================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║               KINEMATION                                  ║
    ║     3D Human Pose Estimation Web Interface                ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Starting server at: http://localhost:8080                ║
    ║                                                           ║
    ║  Pages:                                                   ║
    ║    /         - Home (landing page)                         ║
    ║    /upload   - Video upload + processing                   ║
    ║    /webcam   - Webcam recording + processing               ║
    ║                                                           ║
    ║  API:                                                     ║
    ║    POST /api/upload   - Upload video for processing        ║
    ║    POST /api/webcam   - Upload webcam recording            ║
    ║    GET  /api/status/  - Check processing status            ║
    ║    GET  /api/download/ - Download processed result         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    app.run(host='127.0.0.1', port=8080, debug=False)
