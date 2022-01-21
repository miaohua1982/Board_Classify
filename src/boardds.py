import os
import torch as t
import random
import pandas as pd
from utils import read_image
import cv2

class BoardDatasets:
    def __init__(self, img_base_dir, label_base_dir, transformers, ratio=0.8, is_train=True):
        files_list = os.listdir(img_base_dir)
        random.shuffle(files_list)

        img_list = []
        counter = 0
        for one_file in files_list:
            file_path = os.path.join(img_base_dir,one_file)
            if os.path.isfile(file_path) and one_file[-3:] == 'jpg':
                label_path = os.path.join(label_base_dir,one_file[:-3]+'txt')
                if os.path.exists(label_path):
                    img_list.append([file_path, label_path])
                    counter += 1
        if is_train:
            self.ds_counter = int(counter*ratio)
        else:
            self.ds_counter = counter-int(counter*ratio)

        self.img_list = img_list
        self.transformers = transformers
        self.files_num = counter
        self.train_ratio = ratio
        self.is_train = is_train

    def get_classes_num(self):
        return 2
    
    def __len__(self):
        return self.ds_counter

    def __getitem__(self, idx):
        if self.is_train == False:
            idx = idx + int(self.train_ratio*self.files_num)
        
        img_path = self.img_list[idx][0]
        label_path = self.img_list[idx][1]

        if self.transformers:
            img = read_image(img_path, convert_np=False)
            img = self.transformers(img)
        else:
            img = read_image(img_path, convert_np=True)
            img = t.from_numpy(img)
        
        label = 0
        with open(label_path) as fp:
            lines = fp.readlines()
            for one_line in lines:
                if one_line[0] != '0':
                    label = 1
                    
        return img, label

class BoardDatasets_Alb(BoardDatasets):
    def __init__(self, img_base_dir, label_base_dir, transformers, ratio=0.8, is_train=True):
        super(BoardDatasets_Alb, self).__init__(img_base_dir, label_base_dir, transformers, ratio, is_train)

    def __getitem__(self, idx):
        if self.is_train == False:
            idx = idx + int(self.train_ratio*self.files_num)

        img_path = self.img_list[idx][0]
        label_path = self.img_list[idx][1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #  (224, 224, 3) with np.ndarray
        if self.transformers:
            img = self.transformers(image=img)['image']  # [3, 224, 224]
                
        label = 0
        with open(label_path) as fp:
            lines = fp.readlines()
            for one_line in lines:
                if one_line[0] != '0':
                    label = 1

        return img, label

class BoardDatasets2:
    def __init__(self, train_ds, transformers):
        self.train_ds = train_ds
        self.labels = self.train_ds.label.unique().tolist()
        self.transformers = transformers

    def get_classes_num(self):
        return len(self.labels)
    
    def __len__(self):
        return self.train_ds.shape[0]

    def __getitem__(self, idx):
        one_image = self.train_ds.iloc[idx].tolist()
        img_path = img_path[0]
        if self.transformers:
            img = read_image(img_path, convert_np=False)
            img = self.transformers(img)
        else:
            img = read_image(img_path, convert_np=True)
            img = t.from_numpy(img)
        
        label = one_image[1]
        return img, label

class BoardDatasets_Alb2(BoardDatasets2):
    def __init__(self, train_ds, transformers):
        super(BoardDatasets_Alb2, self).__init__(train_ds, transformers)

    def __getitem__(self, idx):
        one_image = self.train_ds.iloc[idx].tolist()
        img_path = one_image[0]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #  (224, 224, 3) with np.ndarray
        if self.transformers:
            img = self.transformers(image=img)['image']  # [3, 224, 224]
        
        label = one_image[1]
        return img, label


def get_n_splits(img_base_dir, label_base_dir, flod_splits):
    files_list = os.listdir(img_base_dir)
    random.shuffle(files_list)

    img_list = []
    label_list = []
    for one_file in files_list:
        file_path = os.path.join(img_base_dir,one_file)
        if os.path.isfile(file_path) and one_file[-3:] == 'jpg':
            label_path = os.path.join(label_base_dir,one_file[:-3]+'txt')
            if os.path.exists(label_path):
                label = 0
                with open(label_path) as fp:
                    lines = fp.readlines()
                    for one_line in lines:
                        if one_line[0] != '0':
                            label = 1
                img_list.append(file_path)
                label_list.append(label)
    
    train_idxs = []
    test_idxs = []
    for train_idx, test_idx in flod_splits.split(img_list, label_list):
         train_idxs.append(train_idx)
         test_idxs.append(test_idx)
    ds = pd.DataFrame({'img':img_list, 'label':label_list})
    return train_idxs, test_idxs, ds