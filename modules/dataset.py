from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import exposure
from PIL import Image
import cv2
import torch
import numpy as np
import pandas as pd
import os


class SeverityDataset(Dataset):
    def __init__(self, filename_list, label_list, label_df, prefix, global_regression=False,transform=None,augmentation=None,train=True):
        self.filename_list = filename_list
        self.label = label_list
        self.label_df = label_df # pandas dataframe
        self.to_tensor = transforms.ToTensor()
        self.prefix = prefix
        self.train = train
        
        # self.prefix 파일 경로에서 마지막 4번째,3번째 폴더를 제거
        #self.prefix_mask = "/hdd/project/cxr_haziness/data/0407_0608_mask/processed_images/"
        
        self.transform = transform # 이미지 사이즈 변환, 정규화 등 기본 변환
        self.augmentation = augmentation # 온라인 데이터 증강 방법들
        
        self.weak_transform = transforms.Compose([
            
        ])
        
        self.strong_transform = transforms.Compose(
                [
                # Chexpert에서 사용한 augmentation
                
                #transforms.ToPILImage(),
                #transforms.RandomHorizontalFlip(), # 일단 빼고, 나중에 레이블 반영해서 추가하기.
                # distortion
                # 회전
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.8),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.8),
                # random sharpness
                transforms.RandomAdjustSharpness(sharpness_factor=0.0,p=0.8),
                
                # random affine
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05,0.05))], p=0.8),
                
                ]
            )
        
        # global 변수도 추가할 것인가?
        self.global_regression = global_regression
        
        
    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        

        filename = self.filename_list[idx]
        # full_filename이 str이 아니면 변환해준다.
        if type(filename) != str:
            filename = str(filename)        
        
        filename = filename+".png"
        file_path = os.path.join(self.prefix, filename)

        image = Image.open( file_path )
        ##
        
        
        now_label = self.label[idx]
        if type(now_label) != str:
            now_label = str(now_label)
        # label을 ','로 구분하여 리스트로 변환
        now_label = now_label.split(',')
        
        now_label_temp = list(map(int, now_label))
        
        
        now_label = torch.tensor(now_label_temp,dtype=torch.int64)
        now_label_float  = torch.tensor(now_label_temp,dtype=torch.float32)
        
        #augmentation. Albumentation의 데이터 증강은 numpy에만 적용됨.
        
        if self.augmentation:
            image = self.augmentation(image=image)
            image = image['image']
        
        
        # 전처리
        # image = np.array(image)
        # image = exposure.equalize_adapthist(image/255.0)
        ##         
        
        image = self.to_tensor(image)
        
        if self.transform: 
            image = self.transform(image).type(torch.float32)
            if self.train:
                image = self.strong_transform(image)            
        
        # 3채널
        if image.shape[0] != 3:
            image = image.expand(3,-1,-1)
            
        if self.global_regression:
            # global regression을 위한 레이블링. 모든 레이블의 합.
            global_label = torch.tensor([torch.sum(now_label)],dtype=torch.float32)
            return image, now_label, now_label_float, global_label, filename, idx
            
        
        return image, now_label, now_label_float, filename, idx
    