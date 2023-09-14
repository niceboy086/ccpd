# file name check

#from torch.utils.data import *
from imutils import paths
#import cv2
#import numpy as np
#import random
import argparse
from pathlib import Path as pathor


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to the input file")
args = vars(ap.parse_args())
trainDirs = args["images"].split(',')



img_dir = trainDirs
img_paths = []

for i in range(len(img_dir)):
    img_paths += [el for el in paths.list_images(img_dir[i]) if "ccpd_np" not in el]
    # self.img_paths = os.listdir(img_dir)
    # print self.img_paths
#print("随机抽取1000张图片进行测试, *debug, xxx, 20230906")
#sample_num = 1000
#self.img_paths = random.sample(self.img_paths, sample_num)        
        
print("*debug xxx* :", len(img_paths))

for img_name in img_paths:
    print("*debug xxx* : img_name: ", img_name)

    iname = pathor(img_name).name.rsplit('.', 1)[0].split('-')
    print("*debug xxx* : iname: ", iname)
    [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
    print("*debug xxx* : [leftUp, rightDown]: ", [leftUp, rightDown])


'''
for img_name in img_paths:
    print("*debug xxx* : img_name: ", img_name)

    iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')

    img_name = pathor(img_name)
    print("*debug xxx* : img_name: ", img_name)
    print("*debug xxx* : img_name: ", img_name.name)
    img_name = img_name.name
    lbl = img_name.split('.')[0].split('-')[-3]
    print("*debug xxx* : lbl: ", lbl)
'''