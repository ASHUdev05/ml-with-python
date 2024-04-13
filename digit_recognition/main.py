# I have a mnist trained model in the same directory as this file named as mnist_cnn.pt
# I have a test image in the same directory as this file named as test.png
# Use the model to predict the digit in the test image

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

# # Load the test image
# img = np.array(Image.open('test.png').convert('L').resize((28, 28))) / 255.0
# img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

# The images are contained in the test folder read them one by one and predict the digit
# The images are names as 0_1.png, 0_2.png indicating two images of digit 0 and so on
ids = [0,1,2,3,4,5,6,8]
for i in ids:
    # img = np.array(Image.open('test/' + str(i) + '_1.png').convert('L').resize((28, 28))) / 255.0
    # img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
    img = cv2.imread('test/' + str(i) + '_1.png')[:,:,0]
    img = np.invert(np.array(img))
    img = img / 255.0
    img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
    output = model(img)
    print('Prediction:', output.argmax().item())
    print('Confidence:', torch.exp(output).max().item())
    print()
    plt.imshow(img[0][0].numpy(), cmap='gray')
    plt.show()
    print('----------------------')
    print()



    
# img = cv2.imread('test.png')[:,:,0]
# img = np.invert(np.array(img))
# img = img / 255.0
# img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
# output = model(img)
# print('Prediction:', output.argmax().item())
# print('Confidence:', torch.exp(output).max().item())
# print()
# plt.imshow(img[0][0].numpy(), cmap='gray')
# plt.show()
# print('----------------------')
# print()