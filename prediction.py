import torch
from models.Siam_unet import SiamUNet
from torch.autograd import Variable
import utils.dataset as my_dataset
import cv2
import numpy as np
import config.rssia_config as cfg
import preprocessing.transforms as trans
from torch.utils.data import DataLoader
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
    model = SiamUNet()
    if torch.cuda.is_available():
        model.cuda()
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight).items()})
    # model.load_state_dict(torch.load(weight))
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    val_data = my_dataset.Dataset(cfg.VAL_DATA_PATH, cfg.VAL_LABEL_PATH,cfg.VAL_TXT_PATH, 'val', transform=True, transform_med=train_transform_det)
    val_dataloader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    for batch_idx, val_batch in enumerate(val_dataloader):
        model.eval()
        batch_x1, batch_x2, batch_y, _, _, _ = val_batch
        batch_x1, batch_x2, batch_y = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(batch_y).cuda()
        output = model(batch_x1, batch_x2)
        output = torch.sigmoid(output).view(80,80, -1)
        output=output.data.cpu().numpy()#.resize([80, 80, 1])
        output = np.where(output>0.7,255,0)
        #output_final=cv2.merge(output)
        cv2.imwrite('my.png',output)
        # img = Image.fromarray(output,'RGB')
        # img.save('my.png')
        # img.show()
        print("output:",output)
    #
    # pred = model(img1, img2)
    #
    # print(pred)

if __name__ == "__main__":

    weight="weights/model_last.pth"
    img1 = "img_2017/image_2017_960_960_15_0_2.jpg"
    img2 = "img_2018/image_2018_960_960_15_0_2.jpg"
    label = "mask/mask_2017_2018_960_960_15_0_2.jpg"
    prediction(img1, img2, label, weight)