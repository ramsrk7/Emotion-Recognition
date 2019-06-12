
import cv2


def face_func(): #for testing facial recognition
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img = cv2.imread("0.jpg")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

    print(type(faces))

    print(faces)

    a=0
    crop_img = gray_img
    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
        crop_img = img[y:y + h, x:x + w]
        a=a+1



    print(a)
    cv2.imshow("Gray", crop_img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
