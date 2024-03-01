#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from dataloaders import utils
# from scipy.misc import imsave
from utils.Utils import joint_val_image, postprocessing, save_per_img
from utils.metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3_feature import *
import cv2
import numpy as np
from medpy.metric import binary
bceloss = torch.nn.BCELoss()
import imageio
from matplotlib import pyplot as plt
def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    max = np.amax(ent)
    # print(max)

    min = np.amin(ent)
    # print(min)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    '''
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    # save_path = os.path.join(save_root, img_name[0])
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)



def draw_boundary(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]

    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='/media/kd/Seagate Backup Plus Drive/Seagate/DG/code/Deeplabv3p _1027/SCS_deeplabv3/logs_content/test4/lam0.9/20211027_232701.763060/checkpoint_best_66.pth.tar', help='Model path')
    parser.add_argument('--datasetTest', type=list, default=[4], help='test folder id contain images ROIs to test')
    parser.add_argument('--dataset', type=str, default='test', help='test folder id contain images ROIs to test')
    parser.add_argument('-g', '--gpu', type=int, default=0)

    parser.add_argument('--data-dir', default='/media/kd/Seagate Backup Plus Drive/Seagate/DG/dataset/Fundus-doFE/Fundus (copy)', help='data root path')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)
    parser.add_argument('--sync-bn', type=bool, default=False, help='sync-bn in deeplabv3+')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='freeze batch normalization of deeplabv3+')
    parser.add_argument('--movingbn', type=bool, default=False, help='moving batch normalization of deeplabv3+ in the test phase',)
    parser.add_argument('--test-prediction-save-path', type=str, default='./results/saliency/', help='Path root for test image and mask')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest[0]), args.model_file.split('/')[-2])
    os.makedirs(output_path,exist_ok=True)
    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                    transform=composed_transforms_test)
    batch_size = 12
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn,style_mode=False).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    # model_data = torch.load(model_file)

    checkpoint = torch.load(model_file)
    #pretrained_dict = checkpoint
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if args.movingbn:
        model.train()
    else:
        model.eval()




    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):
        data = sample['image']
        target = sample['label']
        img_name = sample['img_name']
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        data.requires_grad_()

        x,_,_,_,_,feature_return= model(data)

        loss=bceloss(torch.sigmoid(x), target)
        loss.backward()
        saliency=abs(data.grad.data)
        saliency,_ = torch.max(saliency,dim=1)
        saliency=saliency.cpu().detach().numpy()
        #feature_return_mean=torch.mean(feature_return,dim=1)
        for i in range(saliency.shape[0]):

            saliency_now=saliency[i]
            saliency_now_max = np.amax(saliency_now)
            saliency_now_min = np.amin(saliency_now)
            saliency_now=((saliency_now-saliency_now_min)/(saliency_now_max-saliency_now_min))*255
            saliency_now=cv2.resize(saliency_now,(800,800))
            root_path=os.path.join(output_path,str(img_name[i]))
            cv2.imwrite(root_path,saliency_now)
            saliency_now=cv2.imread(root_path,flags=1)
            cv2.imwrite(root_path, saliency_now)








if __name__ == '__main__':
    main()
