import cv2

face = cv2.CascadeClassifier('face.xml')
eye = cv2.CascadeClassifier('eye.xml')

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in muka:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0),4)

        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = frame[y:y + h, x:x+w]
        mata = eye.detectMultiScale(roi_gray, 3, 3)
        for (mx,my,mw, mh) in mata:
            cv2.rectangle(roi_warna, (mx, my), (mx+mw, my+mh), (255,255, 0), 3)
    cv2.imshow('Face Detection', frame)
    exit=cv2.waitKey(1) & 0xff
   

cv2.destroyAllWindows()
video.release()