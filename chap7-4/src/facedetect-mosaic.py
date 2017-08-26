import cv2, sys, re

if len(sys.argv) <= 1:
    print("no input file")
    quit()

image_file = sys.argv[1]
output_file = re.sub(r'\.jpg|peg|png$', '-mosaic.jpg', image_file)
face_file = re.sub(r'\.jpg|peg|png$', '-face.jpg', image_file)
mosaic_rate = 30

opencv_path = "/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/"
cascade_file = opencv_path + "haarcascade_frontalface_alt.xml"

image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs,
                                     scaleFactor=1.1,
                                     minNeighbors=1,
                                     minSize=(100,100))

if len(face_list) == 0:
    print("no face")
    quit()

print(face_list)
color = (0,0,255)
for (x, y , w, h) in face_list:
    face_img = image[y:y+h, x:x+w]
    cv2.imwrite(face_file, face_img)
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
    image[y:y+h, x:x+w] = face_img
cv2.imwrite(output_file, image)