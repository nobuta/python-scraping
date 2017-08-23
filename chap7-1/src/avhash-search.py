from PIL import Image
import numpy as np
import os, re

search_dir = "./image/101_ObjectCategories"
cache_dir = "./image/cache_avhash"

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)


def average_hash(fname, size=16):
    # ディレクトリ
    fname2 = fname[len(search_dir):]
    cache_file = cache_dir + "/" + fname2.replace('/', '_') + ".csv"
    if not os.path.exists(cache_file):
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.ANTIALIAS)
        pixel_data = img.getdata()
        pixels = np.array(pixel_data).reshape((size, size))
        avg = pixels.mean()
        px = 1 * (pixels > avg)
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
    else:
        px = np.loadtxt(cache_file, delimiter=",")
    return px


def hamming_dist(a,b):
    aa = a.reshape(1, -1)
    bb = b.reshape(1, -1)
    dist = (aa != bb).sum()
    return dist


def enum_all_files(path):
    for root, dirs, files, in os.walk(path):
        for f in files:
            fname = os.path.join(root, f)
            if re.search(r'\.(jpg|jpeg|png)$', fname):
                yield fname


def find_image(fname, rate):
    src = average_hash(fname)
    for f in enum_all_files(search_dir):
        dst = average_hash(f)
        diff_r = hamming_dist(src, dst) / 256
        # debug
        print("[check] ", f)
        if diff_r < rate:
            yield (diff_r, f)

srcfile = search_dir + "/chair/image_0016.jpg"
html = ""
sim = list(find_image(srcfile, 0.25))
sim = sorted(sim, key=lambda x: x[0])
for r,f in sim:
    print(r, ">", f)
    s = '<div style="float:left;"><h3>[ 差異:' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>' + \
        '<p><a href="' + f + '"><img src="' + f + '" width=400>' + \
        '</a></p></div>'
    html += s

html = """<html><body><h3>元画像</h3><p>
<img src='{0}' width=400></p>{1}
</body></html>""".format(srcfile, html)

with open("./avahash-output.html", "w", encoding="utf-8") as f:
    f.write(html)
print("ok")