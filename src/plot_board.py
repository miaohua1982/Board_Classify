import pandas as pd
from matplotlib import pyplot as plt
import torch as t
import torchvision as tv
from torchvision import transforms
import os
import math
from utils import read_image
import torch_utils as tu
import albumentations
from albumentations import pytorch as AT
import cv2
from utils import read_image
import numpy as np

from boardds import BoardDatasets, BoardDatasets_Alb, BoardDatasets2, BoardDatasets_Alb2, get_n_splits


label_path = "/media/user/myfavor/7.20"
img_base_path = "/media/user/myfavor/7.20"



train_transform = transforms.Compose(
    [transforms.Resize(288),
     transforms.RandomRotation(90),
     #transforms.RandomResizedCrop(224,scale=(0.3,1.0)),
     transforms.RandomAffine(degrees=45, shear=(10, 20, 10, 20), scale=(0.75, 1.2), translate=(0.1, 0.1)),
     transforms.CenterCrop(224),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

test_transform = transforms.Compose(
    [
     transforms.RandomAffine(degrees=45, shear=(10, 30, 10, 30), scale=(0.75, 1.5), translate=(0.1, 0.1)),
     transforms.ToTensor(),
     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

batch_size = 4
trainset = BoardDatasets(img_base_path, label_path, train_transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

x, y = iter(trainloader).next()


# show one picture
plt.figure(1)
col = int(math.sqrt(batch_size))
for i in range(batch_size):
    im = x[i,:,:,:]
    im = im.permute(1,2,0)  # sent channel to last dimesion

    ax1 = plt.subplot(col,col,i+1)
    ax1.axis('off')
    ax1.set_title('flaw' if y[i].item()>0 else 'noflaw')
    ax1.imshow(im, cmap='BrBG')
plt.show()

# show mixup
MIXUP = 0.1
mixup_fn = tu.Mixup(prob=MIXUP, switch_prob=0.0, onehot=True, label_smoothing=0.05, num_classes=trainset.get_classes_num())
x_copy = x.clone()
y_copy = y.clone()
x, y = mixup_fn(x, y)
print(y[0].argmax().item(), ':', y[0].max().item())
print(y_copy[0].item())
plt.figure(1)
mix_one = x[0,:,:,:]
mix_one = mix_one.permute(1,2,0)  # sent channel to last dimesion
plt.imshow(mix_one, cmap='BrBG')
plt.axis('off')
plt.show()

# test different transformer
train_transform = albumentations.Compose([
    albumentations.Resize(224, 224, interpolation=cv2.INTER_AREA),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    #tu.randAugment(N=2,M=6,p=1,cut_out=True),    
    #albumentations.Normalize(),
    AT.ToTensorV2(),    # change to torch.tensor, and permute CHW to HWC
    ])


img1 = cv2.imread(os.path.join(img_base_path, 'Image_20210720125713359.jpg'))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)    #  (224, 224, 3)
img2 = read_image(os.path.join(img_base_path, 'Image_20210720125713359.jpg'))
img2 = np.asarray(img2, dtype=np.float32)


trans_img1 = train_transform(image=img1)['image']  # [3, 112, 112]

plt.figure(1)
# img = x[0,:,:,:].numpy()
img = trans_img1.permute(1,2,0)  # sent channel to last dimesion
plt.imshow(img, cmap='BrBG')
plt.axis('off')
plt.show()