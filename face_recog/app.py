from flask import Flask, redirect, render_template, Response, request
import cv2
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_matched = False

def generate_frames():
    reference_image = cv2.imread('reference.jpg')
    while True:
        success, frame = camera.read()

        try:
            if DeepFace.verify(frame, reference_image.copy())['verified']:
                face_matched = True
            else:
                face_matched = False
        except ValueError:
            face_matched = False

        if not success:
            break
        else:
            if face_matched:
                cv2.putText(frame, 'Face Matched', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Face Not Matched', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    # file.save(secure_filename(file.filename))
    # overwrite reference.jpg with the uploaded image
    with open('reference.jpg', 'wb') as f:
        f.write(file.read())
    # re-render the page
    return redirect('/')


if __name__ == '__main__':
    app.run()