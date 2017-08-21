from sklearn import cross_validation
from PIL import Image
import glob
import numpy as np

caltech_dir = "./image/101_ObjectCategories"
# ここでカテゴリを変更する
categories = ["chair","camera","butterfly","elephant","watch"]
nb_classes = len(categories)

image_w = 64
image_h = 64
# RGB
pixels = image_w * image_h * 3

X = []
Y = []
for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        image = Image.open(f)
        image = image.convert("RGB")
        image = image.resize((image_w, image_h))
        data = np.asarray(image)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

X = np.array(X)
Y = np.array(Y)

#
X_train, X_test, Y_train, Y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)
np.save("./image/5obj.npy", xy)

print("ok", len(Y))
