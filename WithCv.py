import cv2
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from time import *

# To capture video from webcam. 
cap = cv2.VideoCapture(1)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
lis=[]
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(gray)
    image = image.convert('RGB')
    
    pixels = asarray(image)
    detector = MTCNN()
    for results in detector.detect_faces(pixels):
        x,y,w,h=results['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        x1,y1=abs(x),abs(y)
        x2, y2 = x1 + w, y1 + h
        face=pixels[y1:y2,x1:x2]
        image=Image.fromarray(face)
        image=image.resize((160,160))
        face_array=asarray(image)
        lis.append(face_array)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(10) & 0xff
    if k==27:
        
        break
cap.release()
cv2.destroyAllWindows() 
# Release the VideoCapture object


sleep(5)
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
 
# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]
model = load_model('facenet_keras.h5', compile=False)
print('Loaded Model')
testX = list()
for face_pixels in lis:
    embedding = get_embedding(model, face_pixels)
    testX.append(embedding)
testX = asarray(testX)

from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from collections import *

data = load('faces-embeddings.npz')
trainX, trainy= data['arr_0'], data['arr_1']
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
Found=defaultdict(int)
for i in range(len(lis)):
    selection=i
    random_face_pixels = lis[selection]
    random_face_emb = testX[selection]
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    Found[predict_names[0]]+=class_probability
new=[[i,j] for i,j in Found.items()]
new.sort(key=lambda x:x[1],reverse=True)
for i in new:
    print(i)











