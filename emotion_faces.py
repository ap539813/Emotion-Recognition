import numpy as np
import cv2 as cv
from keras.models import load_model

def convert_dtype(x):
    x_float = x.astype('float32')
    return x_float

def normalize(x):
    x_n = (x - 0)/(255)
    return x_n


def reshape(x):
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r
colors = {'neutral':(255, 255, 255), 'angry':(0, 0, 255), 'disgust':(0, 139, 139), 'fear':(0, 0, 0), 'happy':(0, 255, 255), 'sad':(255, 0, 0), 'surprised':(255, 245, 0)}

imotions = {0:'angry', 1:'fear', 2:'happy', 3:'sad',
               4:'surprised', 5:'neutral'}

model = load_model('first_5322_model.hdf5')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)
while True:
    img = cam.read()[1]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation = cv.INTER_AREA)
        roi_gray = convert_dtype(np.array([roi_gray]))
        roi_gray = normalize(roi_gray)
        roi_gray = reshape(roi_gray)
        pr = model.predict(roi_gray)[0]
        print(pr)
        max_emo = np.argmax(pr)
        cv.rectangle(img,(x,y),(x+w,y+h), colors[imotions[max_emo]], 1)
        cv.rectangle(img,(x,y),(x+w,y+h+125), colors[imotions[max_emo]], 1)
        counter = 0
        for i in range(len(pr)):
            cv.rectangle(img, (x, y+h+counter+5), (x + int(w * pr[i]), y+h+counter+20), colors[imotions[i]], -2)
            
            counter += 20
            cv.putText(img, str(int(pr[i]*100)), (x + int(w * pr[i]), (y + h +counter+5)), cv.FONT_HERSHEY_SIMPLEX, 0.50,(0, 0, 0) , 1)
            if i != 5:
                cv.putText(img, imotions[i], (x, (y + h +counter)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(255, 255, 255) , 1)
            else:
                cv.putText(img, imotions[i], (x, (y + h +counter)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 0) , 1)
            
        #cv.circle(img, ((x + w//2), (y + h//2)), int(((h*h + w*w)**0.5)//2), colors[imotions[pr]], 2)
        #cv.putText(img, imotions[pr], ((x + w//2), (y + h//2) - int(((h*h + w*w)**0.5)//2)), cv.FONT_HERSHEY_SIMPLEX, 1, colors[imotions[pr]], 1)
    
    cv.imshow('img',img)
    keypress = cv.waitKey(1)
    if keypress == ord('q'):
        cv.destroyAllWindows()
        break
