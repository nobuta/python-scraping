from PIL import Image
import numpy as np;


def average_hash(fname, size=16):
    img = Image.open(fname)
    img = img.convert('L')
    img = img.resize((size, size), Image.ANTIALIAS)
    pixel_data = img.getdata()
    pixels = np.array(pixel_data)
    pixels = pixels.reshape((size, size))
    avg = pixels.mean()
    diff = 1 * (pixels > avg)
    return diff


def np2hash(n):
    bhash = []
    for nl in ahash.tolist():
        s1 = [str(i) for i in nl]
        s2 = "".join(s1)
        i = int(s2, 2)
        bhash.append("%04x" % i)
    return "".join(bhash)

ahash = average_hash('image_0001.jpg')
print(ahash)
print(np2hash(ahash))