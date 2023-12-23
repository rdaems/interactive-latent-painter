from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import base64
from painter import Painter

app = Flask(__name__)
socketio = SocketIO(app)

buffer = []
processing = False
painter = Painter()

def process_strokes():
    global buffer, processing
    processing = True
    strokes, buffer = buffer, []
    paint(strokes)

    if len(buffer) > 0:
        process_strokes()
    else:
        processing = False

def paint(strokes):
    global painter
    canvas = painter.draw(strokes)
    image = Image.fromarray(canvas)
    byte_array = BytesIO()
    image.save(byte_array, format='PNG')
    byte_array.seek(0)
    data = base64.b64encode(byte_array.getvalue()).decode('utf-8')
    emit('image_updated', data)

@socketio.on('update_strokes')
def update(new_stroke):
    global buffer, processing
    buffer.append(new_stroke)

    if not processing:
        process_strokes()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
