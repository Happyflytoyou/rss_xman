import torch
from models.Siam_unet import SiamUNet
from torch.autograd import Variable
import utils.dataset as my_dataset
import cv2
import numpy as np
import config.rssia_config as cfg
import preprocessing.transforms as trans
from torch.utils.data import DataLoader
from preprocessing.crop_img import splitimage
from PIL import Image

# def img_process(img1, img2, lbl, flag='test'):
#     img1 = img1[:, :, ::-1]  # RGB -> BGR
#     img1 = img1.astype(np.float64)
#     img1 -= cfg.T0_MEAN_VALUE
#     img1 = img1.transpose(2, 0, 1)
#     img1 = torch.from_numpy(img1).float()
#     img2 = img2[:, :, ::-1]  # RGB -> BGR
#     img2 = img2.astype(np.float64)
#     img2 -= cfg.T1_MEAN_VALUE
#     img2 = img2.transpose(2, 0, 1)
#     img2 = torch.from_numpy(img2).float()
#     if flag != 'test':
#         lbl = np.expand_dims(lbl, axis=0)
#         lbl = torch.from_numpy(np.where(lbl > 128, 1.0, 0.0)).float()
#     return img1,img2
def prediction(img1, img2, label, weight):
    print("weight")

    best_metric = 0
    train_transform_det = trans.Compose([
        trans.Scale(cfg.TRANSFROM_SCALES),
    ])
    val_transform_det = trans.Compose([
        trans.Scale(cfg.TRANSFROM_SCALES),
    ])
    test_transform_det = trans.Compose([
        trans.Scale(cfg.TEST_TRANSFROM_SCALES),
    ])
    model = SiamUNet(in_ch=4)
    model=torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight).items()})
    # model.load_state_dict(torch.load(weight))
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    test_data = my_dataset.Dataset(cfg.TEST_DATA_PATH, '',cfg.TEST_TXT_PATH, 'test', transform=True, transform_med=test_transform_det)
    test_dataloader = DataLoader(test_data, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    crop = 0

    rows = 12
    cols = 12
    i = 0
    for batch_idx, val_batch in enumerate(test_dataloader):
        model.eval()
        batch_x1, batch_x2, _, _, h, w = val_batch
        if crop:
            outputs = np.zeros((cfg.TEST_BATCH_SIZE,1,960, 960))

            while (i + w // rows <= w):
                j = 0
                while (j + h // cols <= h):
                    batch_x1_ij = batch_x1[batch_idx, :, i:i + w // rows, j:j + h // cols]
                    batch_x2_ij = batch_x2[batch_idx, :, i:i + w // rows, j:j + h // cols]
                    # batch_y_ij = batch_y[batch_idx,: , i:i + w // rows, j:j + h // cols]
                    batch_x1_ij = np.expand_dims(batch_x1_ij, axis=0)
                    batch_x2_ij = np.expand_dims(batch_x2_ij, axis=0)
                    batch_x1_ij, batch_x2_ij = Variable(torch.from_numpy(batch_x1_ij)).cuda(), Variable(
                        torch.from_numpy(batch_x2_ij)).cuda()
                    with torch.no_grad():
                        output = model(batch_x1_ij, batch_x2_ij)
                    output_w, output_h = output.shape[-2:]
                    output = torch.sigmoid(output).view(-1, output_w, output_h)

                    output = output.data.cpu().numpy()  # .resize([80, 80, 1])
                    output = np.where(output > cfg.THRESH, 255, 0)
                    outputs[batch_idx, :, i:i + w // rows, j:j + h // cols] = output

                    j += h // cols
                i += w // rows

            cv2.imwrite("my_crop.png",outputs[batch_idx,0,:,:])
        else:
            with torch.no_grad():
                output = model(batch_x1, batch_x2)
            output_w, output_h = output.shape[-2:]
            output = torch.sigmoid(output).view(output_w, output_h, -1)
            output = output.data.cpu().numpy()  # .resize([80, 80, 1])
            output = np.where(output > cfg.THRESH, 255, 0)
            # output_final=cv2.merge(output)
            cv2.imwrite('my.png', output)
        # img = Image.fromarray(outputs[batch_idx,:,:,:])
        # img.save('my.png')
        # img.show()
        # print("output:",output)
    #
    # pred = model(img1, img2)
    #
    # print(pred)
# img_2017/image_2017_960_960_10.png img_2018/image_2018_960_960_10.png
if __name__ == "__main__":

    weight="weights/model_last.pth"
    img1 = "img_2017/image_2017_960_960_15_0_2.jpg"
    img2 = "img_2018/image_2018_960_960_15_0_2.jpg"
    label = "mask/mask_2017_2018_960_960_15_0_2.jpg"
    prediction(img1, img2, label, weight)