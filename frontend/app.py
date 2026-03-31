"""
Kinemation Frontend - Flask Application
========================================
Flask-based frontend for the Kinemation 3D pose estimation system.
Provides full HTML/CSS control for pixel-perfect UI recreation.

Usage:
    python app.py

The app will start at http://localhost:8080
"""

from flask import Flask, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__, 
            template_folder='templates',
            static_folder='assets',
            static_url_path='/assets')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║               KINEMATION FRONTEND                          ║
    ║     3D Human Pose Estimation Web Interface                 ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Starting server at: http://localhost:8080                 ║
    ║                                                            ║
    ║  Pages:                                                    ║
    ║    /         - Home (landing page)                         ║
    ║    /upload   - Video upload flow                           ║
    ║    /webcam   - Webcam recording flow                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host='127.0.0.1', port=8080, debug=True)
