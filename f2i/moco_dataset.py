import cv2
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class moco_Dataset(Dataset):
    def __init__(self,moco_pretrained_model):
        self.transform_diff = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(512,interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.transform_moco = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.img_dir = '/data/LargeData/Large/ImageNet/val'
        self.img_dir_list = os.listdir(self.img_dir)
        self.img_dir_list1 = []
        for dir in self.img_dir_list:
            img_dir_list = os.listdir(self.img_dir + '/' + dir)
            self.img_dir_list1.append(img_dir_list)
        self.moco_pretrained_model = moco_pretrained_model
        
    def __len__(self):
        return 50000
    
    def __getitem__(self,idx):
        cate = idx // 50
        pic = idx - cate * 50
        image_path = self.img_dir + '/' + self.img_dir_list[cate] + '/' +self.img_dir_list1[cate][pic]

        # p = Image.open(image_path)
        p = cv2.imread(image_path)

        p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

        p_diff = self.transform_diff(p)
        p_moco = self.transform_moco(p)

        p_moco = torch.unsqueeze(p_moco, dim = 0)

        with torch.no_grad():
            p_feature = self.moco_pretrained_model(p_moco.cuda())

        p_feature = torch.nn.functional.normalize(p_feature, dim=-1)
        p_feature = p_feature.transpose(1,0)
        p_feature = torch.unsqueeze(p_feature,dim = 2)
        # 要变成512,512,3的shape 原来是3,512,512
        p_diff = p_diff.permute(1,2,0)
        # print(type(p_diff))
        # exit()
        return dict(jpg=p_diff, feature=p_feature ,txt ='a high quality detailed professional image')
    
    
