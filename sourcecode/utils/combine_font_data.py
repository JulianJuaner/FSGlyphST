import numpy as np
import cv2
import os
import random

ROOT_DIR = './data/PNGfont_cn'
out_dir = './data/font_combine/data_rand_all'
os.makedirs(out_dir, exist_ok=True)
file_list = os.listdir(ROOT_DIR)
random.shuffle(file_list)

print(file_list)
for i in range(len(file_list)):
    print('processing %d: %s to %s'%(i, file_list[i], file_list[(i+1)%len(file_list)]))
    source = os.path.join(ROOT_DIR, file_list[i])
    target = os.path.join(ROOT_DIR, file_list[(i+1)%len(file_list)])
    source_list = os.listdir(source)
    target_list = os.listdir(target)
    for j in range(1000):
        img1 = cv2.imdecode(np.fromfile(os.path.join(source, source_list[j]), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imdecode(np.fromfile(os.path.join(target, target_list[j]), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        img1 = 255-img1[:,:,3]
        img2 = 255-img2[:,:,3]
        
        cv2.imwrite(os.path.join(out_dir, "%d_%04d.png" % (i, j)), np.hstack((img1, img2)))

def process():
    for file_path in file_list:
        img1 = cv2.imread(os.path.join(source, file_path), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(os.path.join(target, file_path), cv2.IMREAD_UNCHANGED)
        # print(np.unique(img1))
        cv2.imwrite(os.path.join(out_dir, file_path), np.hstack((img1, img2)))