import torch
import argparse
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import shutil
from torchvision import utils as vutils
from cldm.model import create_model,load_state_dict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from f2i.moco_dataset import moco_Dataset
from cldm.logger import ImageLogger
from f2i import vits
import random
import os
import matplotlib.pyplot as plt
import numpy as np


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--use-projector', action='store_true', default=False)

parser.add_argument('--file_path', default=False)



def main():
    args = parser.parse_args()

    # 加载sd+control
    resume_path = './models/control_sd15_f2i_decoder.ckpt'
    model = create_model('./models/cldm_v15_f2idecoder.yaml')
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model = model.cuda()

    model.learning_rate = 1e-5
    model.sd_locked = True
    # 加载moco_pretrained
    moco_pretrained_model = load_moco_pretrained_model(args)
    moco_pretrained_model.to('cuda')
    moco_pretrained_model.eval()
    logger_freq = 300



    dataset = moco_Dataset(moco_pretrained_model=moco_pretrained_model)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],max_epochs=5)

    # Train!
    trainer.fit(model, dataloader)
 




######################################################
######################################################
######################################################
######################################################
# the following are utilities

def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(torch.nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(torch.nn.BatchNorm1d(dim2))
            mlp.append(torch.nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(torch.nn.BatchNorm1d(dim2, affine=False))
    return torch.nn.Sequential(*mlp)

def load_moco_pretrained_model(args):
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch]()
        if args.use_projector:
            hidden_dim = model.head.weight.shape[1]
            del model.head
            model.head = _build_mlp(3, hidden_dim, 4096, 256)
        else:
            model.head = torch.nn.Identity()
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[args.arch]()
        if args.use_projector:
            hidden_dim = model.fc.weight.shape[1]
            del model.fc
            model.fc = _build_mlp(2, hidden_dim, 4096, 256)
        else:
            model.fc = torch.nn.Identity()
        linear_keyword = 'fc'

    if 'https' in args.pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(args.pretrained, map_location="cpu", progress=True, check_hash=False)
    else:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if args.use_projector:
            if k.startswith('module.base_encoder'):
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        else:
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    return model

def build_imagenet_train_loader(args):
    dataset = datasets.ImageFolder('/data/LargeData/Large/ImageNet/train/' , Transform())
    # todo, you need to add a sampler when using distributed data parallel
    # 是这么加吗
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=None, shuffle=True, )
    return loader

class Transform:
    def __init__(self):
        # todo: revise the transform for diffusion models
        # 224*224 picture
        # stable diffusion model and controlnet use 512*512 to train
        # the diffusion model and the controlnet model will convert the picture to 64*64
        # controlnet use a network to convert it 
        self.transform_self = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])
        self.transform_diffusion = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        # self.transform_diffusion = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize()
        # ])
        # this is the transform for moco, but we just need the cosine similarity instead of the feature map
        self.transform_moco = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x_self = self.transform_self(x)
        y1 = self.transform_diffusion(x)
        y2 = self.transform_moco(x)
        return x_self, y1, y2
    

if __name__ == '__main__':
    main()