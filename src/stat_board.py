import pandas as pd
import os
import random

def get_dataset(img_base_dir, label_base_dir):
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
    
    ds = pd.DataFrame({'img':img_list, 'label':label_list})
    return ds


label_base_path = "/media/user/myfavor/7.20"
img_base_path = "/media/user/myfavor/7.20"

train_ds = get_dataset(img_base_path, label_base_path)
count_label = train_ds.groupby('label').count()
labels = train_ds.label.unique().tolist()
print(train_ds.info(verbose=True))
print(train_ds.label.unique().shape)
print(count_label)
print(count_label.describe())