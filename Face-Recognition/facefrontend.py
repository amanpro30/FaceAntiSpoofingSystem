from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from utils import threshold
from io import BytesIO
import json
import random
import cv2
from torchvision import models
import numpy as np
import torch
import torch.nn as nn
from keras.preprocessing import image

model = torch.load("FaceAntiSpoofing.pt")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        model.eval()
        with torch.no_grad():
            pred = model(torch.Tensor(img_array[0]).view(1, 3, 224, 224))
        name="None matching"    
        pred = abs(pred[0][1]).item()  
        if(pred < threshold):
            name='real'
            print(name)
        if(pred >= threshold):
            name='spoofed'
            print(name)
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        print('No face found')
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
