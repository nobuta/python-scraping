import os, glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2, random

image_size = 28
num_max = 20
X = []
Y = []

ttf_list = glob.glob("/Library/Fonts/*.ttf")
ttf_list += glob.glob("~/Library/Fonts/*.ttf")
print("font count=", len(ttf_list))


def draw_text(im, font, text):
    dr = ImageDraw.Draw(im)
    im_size = np.array(im.size)
    fo_size = np.array(font.getsize(text))
    xy = (im_size - fo_size) / 2
    # print(im_size, fo_size)
    dr.text(xy, text, font=font, fill=255)

if not os.path.exists("./image/num"): os.mkdir("./image/num")


def gen_image(base_im, no, font_name):
    for ang in range(-20, 20, 2):
        sub_im = base_im.rotate(ang)
        data = np.asarray(sub_im)
        X.append(data)
        Y.append(no)
        w = image_size
        for r in range(8, 15, 3):
            size = round((r/10) * image_size)
            im2 = cv2.resize(data, (size, size), cv2.INTER_AREA)
            data2 = np.asarray(im2)
            if image_size > size:
                x = (image_size - size) // 2
                data = np.zeros((image_size, image_size))
                data[x:x+size, x:x+size] = data2
            else:
                x = (size - image_size) // 2
                data = data2[x:x+w, x:x+w]
            X.append(data)
            Y.append(no)
            if random.randint(0, 400) == 0:
                fname = "imgae/num/n-{0}-{1}-{2}.png".format(
                    font_name, no, ang, r
                )
                cv2.imwrite(fname, data)

for path in ttf_list:
    print("path:", path)
    font_name = os.path.basename(path)
    try:
        fo = ImageFont.truetype(path, size=100)
    except:
        continue
    for no in range(num_max):
        im = Image.new("L", (200, 200))
        draw_text(im, fo, str(no))
        ima = np.asarray(im)
        blur = cv2.GaussianBlur(ima, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: continue
            num = ima[y:y+h, x:x+w]
            ww = w if w > h else h
            wx = (ww - w) // 2
            wy = (ww - h) // 2
            spc = np.zeros((ww, ww))
            spc[wy:wy+h, wx:wx+w] = num
            num = cv2.resize(spc, (image_size, image_size), cv2.INTER_AREA)
            X.append(num)
            Y.append(no)
            base_im = Image.fromarray(np.uint8(num))
            gen_image(base_im, no, font_name)

print("END: gen_image")
print(len(X))
print(len(Y))
while len(X) % num_max > 0:
    X.pop()
while len(Y) % num_max > 0:
    Y.pop()
print(len(X))
print(len(Y))
X = np.array(X)
print(X)
print(Y)
Y = np.array(Y)
np.savez("./image/font_draw.npz", x=X, y=Y)
print("ok,", len(Y))
