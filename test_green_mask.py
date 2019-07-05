import torch
from models.Siam_unet import SiamUNet
from models.final_Siam_unet import finalSiamUNet
from torch.autograd import Variable
import utils.dataset as my_dataset
import cv2
import numpy as np
import config.rssia_config as cfg
import preprocessing.transforms as trans
from torch.utils.data import DataLoader
from utils.eval import eval_cal
import gdal
from preprocessing.crop_img import splitimage
from PIL import Image

def prediction(weight):
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
    model = SiamUNet()
    # model=torch.nn.DataParallel(model)


    if torch.cuda.is_available():
        model.cuda()
        print('gpu')


    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight).items()})
    # model.load_state_dict(torch.load(weight))
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])
    test_data = my_dataset.Dataset(cfg.TEST_DATA_PATH, cfg.TEST_LABEL_PATH,cfg.TEST_TXT_PATH, 'val', transform=True, transform_med=test_transform_det)
    test_dataloader = DataLoader(test_data, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    crop = 0

    rows = 12
    cols = 12
    i = 0
    for batch_idx, val_batch in enumerate(test_dataloader):
        model.eval()

        batch_x1, batch_x2, mask, im_name, h, w = val_batch
        print('mask_type{}'.format(mask.type))


        with torch.no_grad():
            batch_x1,batch_x2=Variable((batch_x1)).cuda(),Variable(((batch_x2))).cuda()

            try:
                print('try')
                output = model(batch_x1, batch_x2)
                del batch_x1, batch_x2
            except RuntimeError as exception:
                if 'out of memory' in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda,'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print('exception')
                    raise exception
        # print(output)
        output_w, output_h = output.shape[-2:]
        output = torch.sigmoid(output).view(output_w, output_h, -1)
        # print(output)
        output = output.data.cpu().numpy()  # .resize([80, 80, 1])
        output = np.where(output > cfg.THRESH, 255, 0)
        # print(output)
        # have no mask so can not eval_cal
        # precision,recall,F1=eval_cal(output,mask)
        # print('precision:{}\nrecall:{}\nF1:{}'.format(precision,recall,F1))

        print(im_name)
        im_n=im_name[0].split('/')[1].split('.')[0].split('_')
        im__path='final_result/weight50_dmc/mask_2017_2018_960_960_'+im_n[4]+'.tif'


        # im__path = 'weitht50_tif.tif'
        im_data=np.squeeze(output)
        print(im_data.shape)
        im_data=np.array([im_data])
        print(im_data.shape)
        im_geotrans=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        im_proj=''
        im_width=960
        im_height=960
        im_bands=1
        datatype = gdal.GDT_Byte
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(im__path,im_width, im_height, im_bands, datatype)
        if dataset != None:
            print("----{}".format(im__path))
            dataset.SetGeoTransform(im_geotrans)
            dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset

if __name__ == "__main__":

    # weight="model_tif_50.pth"
    # weight="weights/model50.pth"
    weight="weights/model50.pth"
    prediction(weight)