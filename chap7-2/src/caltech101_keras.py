from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

# ここでカテゴリを変更する
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)

image_w = 64
image_h = 64

X_train, X_test, Y_train, Y_test = np.load("./image/5obj.npy")
X_train = X_train.astype("float")
X_test = X_test.astype("float")
print('X_train shape:', X_train.shape)

model = Sequential()
model.add(Convolution2D(32,3,3,
                        border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation("softmax"))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

hdf5_file = "./image/5obj-model.hdf5"
if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)
    print("[Load]", hdf5_file)
else:
    model.fit(X_train, Y_train, batch_size=32, epochs=10)
    model.save_weights(hdf5_file)
    print("[Save]", hdf5_file)

score = model.evaluate(X_test, Y_test)
print('loss=', score[0])
print('accuracy=', score[1])

pre = model.predict(X_test)
for i, v in enumerate(pre):
    pre_ans = v.argmax()
    ans = Y_test[i].argmax()
    dat = X_test[i]
    if ans == pre_ans: continue
    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)
    fname = "" \
            "image/error/" + str(i) + "-" + categories[pre_ans] + \
        "-ne-" + categories[ans] + ".PNG"
    dat *= 256
    img = Image.fromarray(np.uint8(dat))
    img.save(fname)