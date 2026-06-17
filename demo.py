import cv2
import numpy as np
import tensorflow as tf
import json

model = tf.keras.models.load_model("emotion_model_savedmodel", compile=False)
print("Model loaded")

with open("model/class_indices.json") as f2:
    idx = json.load(f2)
labels = {v:k for k,v in idx.items()}
print("Labels:", labels)

cc = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
COLORS={"angry":(0,0,255),"fear":(128,0,128),"happy":(0,200,0),"neutral":(0,200,200),"sad":(255,100,0),"surprise":(0,165,255)}

cap=cv2.VideoCapture(0)
print("Press Q to quit")
while True:
    ret,frame=cap.read()
    if not ret: break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=cc.detectMultiScale(gray,1.3,5,minSize=(48,48))
    for (x,y,w,h) in faces:
        face=cv2.resize(gray[y:y+h,x:x+w],(48,48)).astype("float32")/255.0
        preds=model.predict(face.reshape(1,48,48,1),verbose=0)[0]
        e=labels[np.argmax(preds)]
        c=preds[np.argmax(preds)]*100
        col=COLORS.get(e,(255,255,255))
        cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
        cv2.rectangle(frame,(x,y-35),(x+w,y),col,-1)
        cv2.putText(frame,f"{e.upper()} {c:.1f}%",(x+4,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,f"Faces:{len(faces)} Q=quit",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
    cv2.imshow("Emotion Recognition",frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break
cap.release()
cv2.destroyAllWindows()
