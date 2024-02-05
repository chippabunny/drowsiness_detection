from keras.models import load_model

import cv2
import numpy as np

from PIL import Image



#dataset link:https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset



model = load_model('model_architecture.h5')
faceCascade = cv2.CascadeClassifier("haarcascades_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)


def detect(file_path):
    drowsiness="you are in good condition"
    image_size = (80, 80)
    img = Image.open(file_path).resize(image_size)
    img = np.array(img)/255.0
    result = model.predict(img[np.newaxis, ...])
    print(result)
    predicted_label_index = np.argmax(result)

    if (predicted_label_index==0):
        print(' eyes are closed')
        
    elif predicted_label_index==1:
        print('No Yawn Detected')
        
    elif predicted_label_index==2:
        print(' eyes are opened')
        
    elif predicted_label_index==3:
        print('Yawn Detected')




    if (predicted_label_index==0) & (predicted_label_index==3):
        print('Drowsiness Detected')
        drowsiness="Alert! drowsiness detected"

    elif (predicted_label_index==3):
        print('Drowsiness Detected')
        drowsiness="Alert! drowsiness detected"

    elif predicted_label_index==2 & (predicted_label_index==3):
        print('Drowsiness Detected')
        drowsiness="Alert! drowsiness detected"

    elif (predicted_label_index==1) & (predicted_label_index==2):
        print('No Drowsiness Detected')
        
    elif predicted_label_index==1:
        print('No Drowsiness Detected')
    return drowsiness



wait=0
drowsiness=""
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    key=cv2.waitKey(100)
    wait+=100


    if key==ord('q'):
        break
    if wait==5000:
        filename="frame"+".jpg"
        cv2.imwrite(filename,frame)
        drowsiness=detect(filename)
        wait=0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, drowsiness, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
