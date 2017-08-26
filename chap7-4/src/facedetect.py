import cv2
import os

opencv_path = "/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/"
cascade_file = opencv_path + "haarcascade_frontalface_alt.xml"

def detect(image_file):
    image = cv2.imread(image_file)
    image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_file)

    face_list = cascade.detectMultiScale(image_gr,
                                         scaleFactor=1.1,
                                         minNeighbors=1,
                                         minSize=(150,150))

    if len(face_list) > 0:
        color = (0, 0, 255)
        for face in face_list:
            x,y,w,h = face
            cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=2)
        write_file = os.path.join(
            os.path.dirname(image_file),
            "detect",
            os.path.basename(image_file)
        )
        print("write",write_file)
        if not os.path.exists(os.path.dirname(write_file)):
            os.mkdir(os.path.dirname(write_file))
        cv2.imwrite(write_file, image)
    else:
        print("no face")

files = os.listdir("./image")
for file in files:
    file = os.path.join("./image", file)
    if os.path.isfile(file):
        print("read",file)
        detect(file)