# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-jungong
File Name: window.py.py
Author: chenming
Create Date: 2021/11/8
Description：图形化界面，可以检测摄像头、视频和图片文件
-------------------------------------------------
"""
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch as t
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from model_board import model_board_py, model_board_timm
import albumentations
from albumentations import pytorch as AT

alb_valid_transform = albumentations.Compose([
        albumentations.Resize(672, 672, interpolation=cv2.INTER_AREA),
        albumentations.Normalize(),
        AT.ToTensorV2(),    # change to torch.tensor, and permute CHW to HWC
        ])

# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('Flaw detection system')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        # # 初始化视频读取线程
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.model = self.model_load(weights='/media/user/myfavor/board_weights/model_leaf_2022-01-13_390_0.975610.pkl',
                                     device=self.device)  # todo 指明模型加载的位置的设备
        self.initUI()

    '''
    ***模型初始化***
    '''
    @t.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   classes_num=2,  # classes number
                   ):
        model = model_board_timm(classes_num).to(device)
        net = model.load_state_dict(t.load(weights, map_location=t.device(device)))
        print("模型加载完成!")
        return model

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用目标检测系统\n\n 提供付费指导：有需要的好兄弟加下面的WX即可')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        # label_super.setOpenExternalLinks(True)
        # label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        # about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(about_widget, '联系我')
        self.setTabIcon(0, QIcon('UI/lufei.png'))
        self.setTabIcon(1, QIcon('UI/lufei.png'))
        self.setTabIcon(2, QIcon('UI/lufei.png'))

    '''
    ***上传图片***
    '''
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一放在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("UI/right.jpeg"))

    '''
    ***检测图片***
    '''
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = 640  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            img = cv2.imread(source)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = alb_valid_transform(image=img)['image']  # [3, h, w]
            print(type(img), img.shape)
            img = img.unsqueeze(0)                          # [1, 3, h, w]
            pred_prob = self.model(img)
            pred_prob = pred_prob.softmax(dim=1)
            print('The detect prob is', pred_prob)
            pred_prob, pred_label = pred_prob.max(dim=1)
            pred_label = pred_label.cpu().item()
            pred_prob = pred_prob.cpu().item()
            # read gt
            label_path = source[:source.rfind('.')]+'.txt'
            print(label_path)
            if os.path.exists(label_path):
                label = 0
                with open(label_path) as fp:
                    lines = fp.readlines()
                    for one_line in lines:
                        if one_line[0] != '0':
                            label = 1
                gt_msg = "实际标签：有瑕疵" if label > 0 else "实际标签：无瑕疵"
            else:
                gt_msg = "标签文件丢失"
            msg = '有瑕疵，概率为%.2f%%，%s' % (pred_prob*100, gt_msg) if pred_label > 0 else '无瑕疵，概率为%.2f%%，%s'%(pred_prob*100, gt_msg)
            # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
            if pred_label > 0:
                self.right_img.setPixmap(QPixmap("UI/flaw.jpg"))
            else:
                self.right_img.setPixmap(QPixmap("UI/noflaw.jpg"))
            QMessageBox.warning(self, "检测结果", msg)

    '''
    ### 界面关闭事件 ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
