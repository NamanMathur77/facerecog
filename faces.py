import numpy as np
import cv2
import pickle
from firebase import firebase
from datetime import datetime

firebase = firebase.FirebaseApplication('https://facerecog-e2c4b.firebaseio.com/')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("lables.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)
time = datetime.now()

while(True):
	ret, frame = cap.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for(x, y, w, h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] # (cord1-height, cord2-height)
		roi_color = frame[y:y+h, x:x+w]
		id_, conf = recognizer.predict(roi_gray)
		if conf>=45: #and conf <=85:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, 2)
		img_item = "1.png"
		cv2.imwrite(img_item, roi_color)
		color = (255, 0, 0)#RGB
		stroke = 2
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y),(end_cord_x, end_cord_y), color, stroke)


	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		print(labels[id_])
		n=labels[id_]
		#time=now.strftime("%d/%m/%Y")
		name=time.strftime("%Y-%m-%d")
		t=time.strftime("%H")
		final=name+"/"+t
		additem=firebase.post(final,{n:'p'})

cap.release()
cv2.destroyAllWindows()