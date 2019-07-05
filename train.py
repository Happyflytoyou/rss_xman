from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
from models.Siam_unet import SiamUNet, SiamUNetU
from models.loss import calc_loss, FocalLoss2d, calc_loss_L4
from torch.optim.lr_scheduler import StepLR
import utils.dataset as my_dataset
import config.rssia_config as cfg
import preprocessing.transforms as trans
import os
from tensorboardX import SummaryWriter

def main():
    best_metric = 0
    train_transform_det = trans.Compose([
        trans.Scale(cfg.TRANSFROM_SCALES),
    ])
    val_transform_det = trans.Compose([
        trans.Scale(cfg.TRANSFROM_SCALES),
    ])

    train_data = my_dataset.Dataset(cfg.TRAIN_DATA_PATH, cfg.TRAIN_LABEL_PATH,
                                    cfg.TRAIN_TXT_PATH, 'train', transform=True, transform_med=train_transform_det)
    val_data = my_dataset.Dataset(cfg.VAL_DATA_PATH, cfg.VAL_LABEL_PATH,
                                  cfg.VAL_TXT_PATH, 'val', transform=True, transform_med=val_transform_det)
    train_dataloader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = SiamUNet(in_ch=3)
    if cfg.RESUME:
        checkpoint = torch.load(cfg.TRAINED_LAST_MODEL)
        model.load_state_dict(checkpoint['state_dict'])
        print('resume success \n')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()


    # if torch.cuda.is_available():
    #     model.cuda()

    # params = [{'params': md.parameters()} for md in model.children() if md in [model.classifier]]
    optimizer = optim.Adam(model.parameters(), lr=cfg.INIT_LEARNING_RATE, weight_decay=cfg.DECAY)
    fl = FocalLoss2d(gamma=cfg.FOCAL_LOSS_GAMMA)
    Loss_list = []
    Accuracy_list = []
    scheduler = StepLR(optimizer,step_size=8, gamma=0.1)
    for epoch in range(cfg.EPOCH):
        scheduler.step()
        print('epoch {}'.format(epoch+1))
        #training--------------------------
        train_loss = 0
        train_acc = 0
        for batch_idx, train_batch in enumerate(train_dataloader):
            model.train()
            batch_x1, batch_x2, batch_y, _, _, _ = train_batch
            batch_x1, batch_x2, batch_y = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(batch_y).cuda()
            outputs = model(batch_x1, batch_x2)
            del batch_x1, batch_x2
            loss = calc_loss_L4(outputs[0], outputs[1], outputs[2], outputs[3], batch_y)
            # train_loss += loss.data[0]
            #should change after
            # pred = torch.max(out, 1)[0]
            # train_correct = (pred == batch_y).sum()
            # train_acc += train_correct.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if(batch_idx) % 5 == 0:
                model.eval()
                val_loss = 0
                for v_batch_idx, val_batch in enumerate(val_dataloader):
                    v_batch_x1, v_batch_x2, v_batch_y, _, _, _ = val_batch
                    v_batch_x1, v_batch_x2, v_batch_y = Variable(v_batch_x1).cuda(), Variable(v_batch_x2).cuda(), Variable(
                        v_batch_y).cuda()
                    val_outs = model(v_batch_x1,v_batch_x2)
                    del v_batch_x1, v_batch_x2
                    val_loss += float(calc_loss_L4(val_outs[0], val_outs[1], val_outs[2], val_outs[3],v_batch_y))
                del val_outs, v_batch_y
                print("Train Loss: {:.6f}  Val Loss: {:.10f}".format(loss, val_loss))

        if (epoch+1)%5 == 0:
            torch.save({'state_dict':model.state_dict()},
                       os.path.join(cfg.SAVE_MODEL_PATH, cfg.TRAIN_LOSS, 'model_tif_'+str(epoch+1)+'.pth'))
    torch.save({'state_dict': model.state_dict()},
                   os.path.join(cfg.SAVE_MODEL_PATH, cfg.TRAIN_LOSS, 'model_tif_last.pth'))
if __name__ == '__main__':
    main()