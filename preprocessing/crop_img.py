import os
import cv2
from PIL import Image, ImageFilter

# def splitimage(img_src, row, col, des_path):
#     ext = os.path.splitext(img_src)
#     img = cv2.imread(img_src)
#
#     size = img.shape
#     boxs = []
#     sig_box = [size[0] / col, size[1] / row]
#     i = 0
#     id = 1
#     while (i + sig_box[0] <= size[0]):
#         j = 0
#         while (j + sig_box[1] <= size[1]):
#             new_img = img((i, j, i + sig_box[0], j + sig_box[1]))
#             save_path = os.path.join(des_path, '_' + str(i/sig_box[0])+ '_' + str(j/sig_box[1]) + '.jpg')
#             # cv2.resize(new_img, (224, 224))
#             cv2.imwrite(save_path, new_img)
#             id += 1
#             j += sig_box[1]
#         i += sig_box[0]

def splitimage(img_src, rows, cols, des_path, filename):
    ext = os.path.splitext(img_src)
    img = Image.open(img_src)
    size = img.size
    if(not os.path.exists(des_path)):
        os.mkdir(des_path)
    sig_box = [size[0]/cols, size[1]/rows]
    i = 0
    id = 1
    row = 0
    col = 0
    while (i + sig_box[0] <= size[0]):
        j = 0
        col = 0
        while (j + sig_box[1] <= size[1]):
            new_img = img.crop([i, j, i + sig_box[0], j + sig_box[1]])
            save_path = os.path.join(des_path, filename + '_' + str(row)+ '_' + str(col) + '.jpg')
            j += sig_box[1]
            col += 1
            if(os.path.exists(save_path)):
                continue
            new_img.save(save_path)
            id += 1
        i += sig_box[0]
        row += 1


source_path = "/home/ubuntu/PycharmProjects/datasets/rssrai2019_cd_v2/"
des_path = "/home/ubuntu/PycharmProjects/datasets/rssrai2019_croped/"
imgs_dir = ['img_2017', 'img_2018','mask']
modes = ['train', 'val','test']
for mode in modes:
    source_mode_path = os.path.join(source_path,mode)
    des_mode_path = os.path.join(des_path, mode)

    for img_dir in imgs_dir:
        source_img_path = os.path.join(source_mode_path, img_dir)
        if not os.path.exists(source_img_path):
            continue
        des_img_path = os.path.join(des_mode_path, img_dir)
        imgFileList = os.listdir(source_img_path)
        for filename in imgFileList:
            img_src = os.path.join(source_img_path,filename)
            if img_src.endswith('.jpg') or img_src.endswith('.png'):
                filename = img_src.split("/")[-1].split('.')[0]
                splitimage(img_src,12, 12, des_img_path,filename)




