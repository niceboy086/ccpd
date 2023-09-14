from torch.utils.data import *
from imutils import paths
import cv2
import numpy as np
import random
from pathlib import Path as pathor


class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i]) if "ccpd_np" not in el]
        # self.img_paths = os.listdir(img_dir)
        # print("*debug xxx* :", self.img_paths)
        print("随机抽取2000张图片进行rpnet训练, *debug, xxx, 20230906")
        sample_num = 2000
        self.img_paths = random.sample(self.img_paths, sample_num)        
        
        print("*debug xxx* :", len(self.img_paths))
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        # lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
        lbl = pathor(img_name).name.split('.')[0].split('-')[-3]

        # iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        iname = pathor(img_name).name.rsplit('.', 1)[0].split('-')
        # fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
        # leftUp, rightDown = [min([fps[el][0] for el in range(4)]), min([fps[el][1] for el in range(4)])], [
        #     max([fps[el][0] for el in range(4)]), max([fps[el][1] for el in range(4)])]
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

        return resizedImage, new_labels, lbl, img_name


class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i]) if "ccpd_np" not in el]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        print("随机抽取1000张图片进行测试, *debug, xxx, 20230906")
        sample_num = 1000
        self.img_paths = random.sample(self.img_paths, sample_num)        
        
        print("*debug xxx* :", len(self.img_paths))
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        # print("*debug xxx* : img_name: ", img_name)
        lbl = pathor(img_name).name.split('.')[0].split('-')[-3]
        # print("*debug xxx* : lbl: ", lbl)
        return resizedImage, lbl, img_name



class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir,imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i]) if "ccpd_np" not in el]
        # self.img_paths = os.listdir(img_dir)
        # print(self.img_paths)
        print("随机抽取2000张图片进行训练, *debug, xxx, 20230906")
        sample_num = 2000
        self.img_paths = random.sample(self.img_paths, sample_num)        
        
        print("*debug xxx* :", len(self.img_paths))
        # print("*debug xxx* :", self.img_paths)
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        # print("*debug*:",  img_name)
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        # iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        iname = pathor(img_name).name.rsplit('.', 1)[0].split('-')
        # print("*debug*:",  iname)
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

        # tps = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        # for dot in tps:
        #     cv2.circle(img, (int(dot[0]), int(dot[1])), 2, (0, 0, 255), 2)
        # cv2.imwrite("/home/xubb/1_new.jpg", img)

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [(leftUp[0] + rightDown[0])/(2*ori_w), (leftUp[1] + rightDown[1])/(2*ori_h), (rightDown[0]-leftUp[0])/ori_w, (rightDown[1]-leftUp[1])/ori_h]

        resizedImage = resizedImage.astype('float32')
        # Y = Y.astype('int8')
        resizedImage /= 255.0
        # lbl = img_name.split('.')[0].rsplit('-',1)[-1].split('_')[:-1]
        # lbl = img_name.split('/')[-1].split('.')[0].rsplit('-',1)[-1]
        # lbl = map(int, lbl)
        # lbl2 = [[el] for el in lbl]

        # resizedImage = torch.from_numpy(resizedImage).float()
        return resizedImage, new_labels


class demoTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        return resizedImage, img_name
