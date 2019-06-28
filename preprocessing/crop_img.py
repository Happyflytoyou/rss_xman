import os
import cv2
from PIL import Image, ImageFilter

def splitimage(img_src, row, col, des_path, resize_w, resize_h):
    ext = os.path.splitext(img_src)
    img = Image.open(img_src)
    size = img.size
    if(not os.path.exists(des_path)):
        os.mkdir(des_path)
    sig_box = [size[0]/col, size[1]/row]
    i = 0
    id = 1
    while (i + sig_box[0] <= size[0]):
        j = 0
        while (j + sig_box[1] <= size[1]):
            new_img = img.crop([i, j, i + sig_box[0], j + sig_box[1]]).resize((resize_w, resize_h),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            save_path = os.path.join(des_path, str(id ) + '.jpg')
            j += sig_box[1]
            if(os.path.exists(save_path)):
                continue
            new_img.save(save_path)
            id += 1
        i += sig_box[0]
