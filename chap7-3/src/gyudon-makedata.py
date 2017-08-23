from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np

root_dir = "./image/"
categories = ["normal", "bani", "negi", "cheese"]
nb_classes = len(categories)
image_size = 50

X = []
Y = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    print("-- ", cat, "を処理中")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./image/gyudon.npy", xy)
print("ok", len(Y))