from flask import Flask, render_template
from flask_socketio import SocketIO
from io import BytesIO
from PIL import Image
import base64
from painter import Painter
import threading
import time
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

buffer = []
processing_lock = threading.Lock()
painter = Painter()
painter.encode_text('Pieter Breughel the Elder')

def process_strokes():
    global buffer, processing_lock
    if not processing_lock.locked() and len(buffer) > 0:
        with processing_lock:
            strokes = buffer.copy()
            buffer.clear()
            paint(strokes)

        process_strokes()

def paint(strokes):
    global painter
    canvas = painter.paint(strokes)
    image = Image.fromarray(canvas)
    byte_array = BytesIO()
    image.save(byte_array, format='PNG')
    byte_array.seek(0)
    data = base64.b64encode(byte_array.getvalue()).decode('utf-8')
    socketio.emit('image_updated', data)

@socketio.on('update_strokes')
def update(new_stroke):
    global buffer
    buffer.append(new_stroke)
    socketio.start_background_task(process_strokes)

@socketio.on('update_cursor_size')
def update_cursor_size(size):
    global painter
    painter.patch_radius = float(size) / 2

@socketio.on('update_time_threshold')
def update_time_threshold(threshold):
    global painter
    painter.time_threshold = float(threshold)

@socketio.on('view_timemap')
def view_timemap():
    global painter
    canvas = painter.timemap_to_image()
    image = Image.fromarray(canvas)
    byte_array = BytesIO()
    image.save(byte_array, format='PNG')
    byte_array.seek(0)
    data = base64.b64encode(byte_array.getvalue()).decode('utf-8')
    socketio.emit('image_updated', data)

@socketio.on('view_timemap_end')
def view_timemap_end():
    global painter
    canvas = painter.latents_to_image()
    image = Image.fromarray(canvas)
    byte_array = BytesIO()
    image.save(byte_array, format='PNG')
    byte_array.seek(0)
    data = base64.b64encode(byte_array.getvalue()).decode('utf-8')
    socketio.emit('image_updated', data)

@socketio.on('update_brush')
def update_brush(data):
    global painter
    if data['type'] == 'noise':
        painter.set_brush(noise=True)
    elif data['type'] == 'prompt':
        painter.set_brush(prompt=data['prompt'])

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
