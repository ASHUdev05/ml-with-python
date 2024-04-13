from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
app.config['UPLOAD_FOLDER'] = 'test'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the model
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
prediction, confidence = 0, 0

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    prediction = 0  # Initialize prediction with a default value
    confidence = 0
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        image = cv2.imread('uploads/' + filename)[:,:,0]
        image = np.invert(np.array(image))
        image = image / 255.0
        image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
        output = model(image)
        prediction = output.argmax().item()
        confidence = torch.exp(output).max().item()
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url, prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)