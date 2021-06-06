#=================================================
#==Adapted by  : Sundaresan S                    ==
#==Date        : 5 Jun 2021      			     ==
#==Description : made with pretrained models  	 ==
#==			                                     ==
#==			   			  	                	 ==
#==================================================
import cv2
myface = cv2.CascadeClassifier('haar-cascade-files-master\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar-cascade-files-master\haarcascade_eye.xml')
img1 = cv2.VideoCapture(1)
while True:
    _,img = img1.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = myface.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #roi_gray = gray[y:y + h, x:x + w]
        #roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 4)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    print(faces)
    img=cv2.resize(img, (720, 720))
    cv2.imshow('img', img)
    k = cv2.waitKey(30)&0xff
    if  k==27:
        break
img1.release()


