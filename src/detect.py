import torch as t
from torchvision import transforms
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
import torchvision as tv
import os
import time
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from boardds import BoardDatasets, BoardDatasets_Alb
from model_board import model_board_py, model_board_timm

import torch_utils as tu

import albumentations
from albumentations import pytorch as AT
import cv2

label_path = "/media/user/myfavor/7.20"
img_base_path = "/media/user/myfavor/7.20"


def detect(batch_size, classes_num, model_path):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model = model_board_timm(classes_num).to(device)
    net = model.load_state_dict(t.load(model_path, map_location=t.device(device)))

    alb_valid_transform = albumentations.Compose([
        albumentations.Resize(672, 672, interpolation=cv2.INTER_AREA),
        albumentations.Normalize(),
        AT.ToTensorV2(),    # change to torch.tensor, and permute CHW to HWC
        ])
    
    test_dateset = BoardDatasets_Alb(img_base_path, label_path, alb_valid_transform, ratio=0.95, is_train=True)
    test_dateloader = t.utils.data.DataLoader(test_dateset, batch_size=batch_size, shuffle=True, drop_last=False)

    print('[%s] we ara testing model on test dataset...' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

    gt_equal = 0.0
    pred_pos_counter = 0
    tp_counter = 0
    counter = 0
    gt_arr = []
    pred_prob = []
    for data in test_dateloader:
        inputs, labels = data  # labels: [batch_size, 1]
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # outputs: [batch_size, ]

        # acc
        gt = (outputs.argmax(dim=1) == labels).float().sum()
        counter += outputs.shape[0]
        # recall
        pred_pos = outputs.argmax(dim=1).sum()
        tp = labels & outputs.argmax(dim=1)
        tp = tp.sum()
        # pricision & recall
        gt_equal += gt.cpu().item()
        pred_pos_counter += pred_pos.cpu().item()
        tp_counter += tp.cpu().item()
        # total
        gt_arr.extend(labels.cpu().tolist())
        outputs = outputs.softmax(dim=1)
        pred_prob.extend(outputs[:,1].cpu().tolist())
    
    print('[%s] Finished Training' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

    return  tp_counter/pred_pos_counter, gt_equal/counter, gt_arr, pred_prob

#recall:0.9070, pricision:0.9390
#recall:0.9318, pricision:0.9634
#recall:0.8438, pricision:0.9390
#recall:0.9189, pricision:0.9634

if __name__ == '__main__':
    batch_size = 2
    classes_num = 2
    model_path = '/media/user/myfavor/board_weights/model_leaf_2022-01-13_390_0.975610.pkl'
    r, p, gt_arr, pred_prob = detect(batch_size, classes_num, model_path)
    print('recall:%.4f, precision:%.4f' % (r, p))

    precision, recall, thresholds = precision_recall_curve(gt_arr, pred_prob)
    print('-'*60)
    plt.figure("P-R Curve")
    plt.title("Precision/Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision)
    plt.show()
