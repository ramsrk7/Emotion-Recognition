from __future__ import division
from keras.models import model_from_json
import numpy as np
import cv2


#loading the model


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


#setting image resizing parameters


WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



def videocapture():
    video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    a = 1

    while True:
        a = a+1
        check, frame = video.read()
        frame = cv2.flip(frame,1,0)
        print(check)
        print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)


        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            yhat = loaded_model.predict(cropped_img)
            cv2.putText(frame, labels[int(np.argmax(yhat))], (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 1, cv2.LINE_AA)
            print("Emotion: " + labels[int(np.argmax(yhat))])


        cv2.imshow("Capturing",frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    print(a)
    video.release()
    cv2.destroyAllWindows()


videocapture()






