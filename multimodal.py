# encoding: utf-8

import math
from PIL import Image  #PIL=python image library
import random
import  numpy as np
import random
import cv2



########################### this code is for Local Grayscale Patch Replacement(LGPR)  #################################
class LGPR(object):

    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        new = img.convert("L")
        np_img = np.array(new, dtype=np.uint8)
        img_gray = np.dstack([np_img, np_img, np_img])

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[1] and h < img.size[0]:
                x1 = random.randint(0, img.size[0] - h)
                y1 = random.randint(0, img.size[1] - w)
                img = np.asarray(img).astype('float')

                img[y1:y1 + h, x1:x1 + w, 0] = img_gray[y1:y1 + h, x1:x1 + w, 0]
                img[y1:y1 + h, x1:x1 + w, 1] = img_gray[y1:y1 + h, x1:x1 + w, 1]
                img[y1:y1 + h, x1:x1 + w, 2] = img_gray[y1:y1 + h, x1:x1 + w, 2]

                img = Image.fromarray(img.astype('uint8'))

                return img

        return img
#######################################################################################################################
################################ this code is for Multi-Modal Defense  ################################################

def toSketch(img):  # Convert visible  image to sketch image
    img_np = np.asarray(img)
    img_inv = 255 - img_np
    img_blur = cv2.GaussianBlur(img_inv, ksize=(27, 27), sigmaX=0, sigmaY=0)
    img_blend = cv2.divide(img_np, 255 - img_blur, scale=256)
    img_blend = Image.fromarray(img_blend)
    return img_blend

"""
Randomly select several channels of visible image (R, G, B), gray image (gray), and sketch image (sketch) 
to fuse them into a new 3-channel image.
"""
def random_choose(r, g, b, gray_or_sketch):
    p = [r, g, b, gray_or_sketch, gray_or_sketch]
    idx = [0, 1, 2, 3, 4]
    random.shuffle(idx)
    return Image.merge('RGB', [p[idx[0]], p[idx[1]], p[idx[2]]])


# 10(%Grayscale) 5%(Grayscale-RGB) 5%(Sketch-RGB)
class Fuse_RGB_Gray_Sketch(object):
    def __init__(self,G=0.1,G_rgb = 0.05,S_rgb =0.05):
        self.G = G
        self.G_rgb = G_rgb
        self.S_rgb = S_rgb

    def __call__(self, img):
        r, g, b = img.split()
        gray = img.convert('L') #convert visible  image to grayscale images
        p = random.random()
        if p < self.G: #just Grayscale
            return Image.merge('RGB', [gray, gray, gray])

        elif p < self.G + self.G_rgb: #fuse Grayscale-RGB
            img2 = random_choose(r, g, b, gray)
            return img2

        elif p < self.G + self.G_rgb + self.S_rgb: #fuse Sketch-RGB
            sketch = toSketch(gray)
            img3 = random_choose(r, g, b, sketch)
            return img3
        else:
            return img





