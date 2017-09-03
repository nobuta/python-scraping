from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import numpy as np

image_w = 28
image_h = 28
nb_classes = 20


def main():
    # フォント画像のデータを読む
    xy = np.load("./image/font_draw.npz")
    X = xy["x"]
    Y = xy["y"]
    # データを正規化
    X = X.reshape(X.shape[0], image_w * image_h).astype('float32')
    X /= 255
    print(len(X))
    print(len(X[0]))
    print(len(Y))
    Y = np_utils.to_categorical(Y)
    print(Y[0])
    print(len(Y))
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    print("X_train")
    print(X_train)
    print(len(X_train))
    print(len(X_train[0]))
    print("X_test")
    print(X_test)
    print(len(X_test))
    print(len(X_test[0]))
    print("y_train")
    print(y_train)
    print(len(y_train))
    print(len(y_train[0]))
    print("y_test")
    print(y_test)
    print(len(y_test))
    print(len(y_test[0]))
    # モデルを構築
    model = build_model()
    model.fit(X_train, y_train,
              batch_size=128, epochs=20, verbose=1,
              validation_data=(X_test, y_test))
    # モデルを保存
    model.save_weights('font_draw.hdf5')
    # モデルを評価
    score = model.evaluate(X_test, y_test, verbose=0)
    print('score=', score)

def build_model():
    # MLPのモデルを構築
    model = Sequential()
    model.add(Dense(512, input_shape=(image_w * image_h,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()

