from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
from models.Siam_unet import SiamUNet
from models.loss import calc_loss
import utils.dataset as my_dataset
import config.rssia_config as cfg
import preprocessing.transforms as trans
import os

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

    model = SiamUNet()
    if torch.cuda.is_available():
        model.cuda()

    # params = [{'params': md.parameters()} for md in model.children() if md in [model.classifier]]
    optimizer = optim.Adam(model.parameters(), lr=cfg.INIT_LEARNING_RATE, weight_decay=cfg.DECAY)

    Loss_list = []
    Accuracy_list = []

    for epoch in range(cfg.EPOCH):
        print('epoch {}'.format(epoch+1))
        #training--------------------------
        train_loss = 0
        train_acc = 0
        for batch_x1, batch_x2, batch_y in train_dataloader:
            batch_x1, batch_x2, batch_y = Variable(batch_x1).cuda(), Variable(batch_x2).cuda(), Variable(batch_y).cuda()
            out = model(batch_x1, batch_x2)
            loss = calc_loss(out, batch_y)
            # train_loss += loss.data[0]
            #should change after
            pred = torch.max(out, 1)[0]
            # train_correct = (pred == batch_y).sum()
            # train_acc += train_correct.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Train Loss: {:.6f}, Acc: {:.6f}".format(0, loss))
        if (epoch+1)%5 == 0:
            torch.save({'state_dict':model.state_dict()},
                       os.path.join(cfg.SAVE_MODEL_PATH, 'model'+str(epoch+1)+'.pth'))
if __name__ == '__main__':
    main()