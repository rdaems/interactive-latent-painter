from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image, ImageDraw
import base64
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Placeholder for the strokes received from the frontend
strokes = []

def process_strokes():
    # Simulate processing time
    time.sleep(2)
    
    # Create a new image and draw strokes on it
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    for i in range(len(strokes) - 1):
        draw.line([strokes[i]['x'], strokes[i]['y'], strokes[i + 1]['x'], strokes[i + 1]['y']], fill='black', width=2)
    
    # Save the image to a BytesIO object
    img_byte_array = BytesIO()
    img.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    
    # Encode the image to base64 to send it to the client
    img_data = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    return img_data

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('update_strokes')
def update(message):
    print('update')
    global strokes
    new_strokes = message['strokes']
    strokes.extend(new_strokes)
    img_data = process_strokes()
    emit('image_updated', {'imageData': img_data})

if __name__ == '__main__':
    socketio.run(app, debug=True)
