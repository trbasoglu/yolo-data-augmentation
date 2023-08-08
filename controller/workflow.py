from controller.apply_album_aug import apply_aug
from controller.get_album_bb import get_bboxes_list
import cv2
import os
import yaml
import random 
from tqdm import tqdm
with open("contants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)


def run_pipeline():
    imgs = os.listdir(CONSTANTS["inp_img_pth"])  
    n = int(CONSTANTS["generationNumber"])
    for i in tqdm(range(0,n)):   
        random_index = random.randint(0,len(imgs)-1)
        img_file = imgs[random_index]
        try:
            file_name = img_file.split('.')[0]
            aug_file_name = file_name + "_aug_" + str(i)
            image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))           
            lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], file_name + '.txt')                                
            album_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])
            apply_aug(image, album_bboxes, CONSTANTS["out_lab_pth"],  CONSTANTS["out_img_pth"], aug_file_name, CONSTANTS['CLASSES'])
        except:
            continue