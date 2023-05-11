# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import csv
import random
from matplotlib import pyplot as plt
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN

import pandas as pd


logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(args.device)
    model.eval()
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def test(args, model):
    """ Train the model """

    # Prepare dataset
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    testset = datasets.ImageFolder(os.path.join(args.test_dir), transform=transform_test)

    imgList = [list(testset.imgs[i])[0].split('/')[-1] for i in range(len(testset))]
    #print(imgList)
    
    test_sampler = SequentialSampler(testset)
    
    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True) if testset is not None else None

    test_bar = tqdm(test_loader, desc=f'Testing')
    all_preds, all_label, all_logit = [], [], []
    with torch.no_grad():
        for batch_data in test_bar:
            image, label = batch_data
            image = image.to(device)
            label = label.to(device)
            logits = model(image)[0]
            preds = torch.argmax(logits, dim=-1)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(label.detach().cpu().numpy())
                all_logit.append(logits.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], label.detach().cpu().numpy(), axis=0
                )
                all_logit[0] = np.append(
                    all_logit[0], logits.detach().cpu().numpy(), axis=0
                )
        test_bar.close()

    #print(classification_report(all_label[0], all_preds[0], target_names=[str(i) for i in range(args.num_classes)], digits=6))

    # covariance matrix
    #print(all_preds[0])
    #print(all_preds[0].reshape(-1, 1))
    #print(all_preds[0].reshape(args.num_classes, 2))
    #print(all_label[0].reshape(args.num_classes, 2))
    #print(np.corrcoef(all_preds[0].reshape((args.num_classes, 2)), all_label[0].reshape((args.num_classes, 2))))
    return all_preds[0], all_label[0], all_logit[0], imgList
    


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type",
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for trained ViT models.")
    parser.add_argument("--img_size", default=224, type=str,
                        help="Resolution size")
    parser.add_argument("--test_dir", default='/home/yuhsi/Badminton/src/postprocess/HitFrame/',
                        help="Where to do the inference.")
    parser.add_argument("--dataset", default='',
                        help="What kind of dataset to do the inference.")

    parser.add_argument("--num_classes", default=9, type=int,    ################################ 1. class-num ################################
                        help="Number of classes")

    parser.add_argument("--local_rank", type=int, default=-1,
                            help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)



    ################
    # ensemble
    ################
    def softmax(x):
        
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    # Model & Tokenizer Setup
    models = args.model_type
    models = models[1:-1]
    modelTypes = models.split(',')
    #print(modelTypes)
    #>['ViT-B_16', 'ViT-B_16']

    cp = args.checkpoint
    cp = cp[1:-1]
    cpFiles = cp.split(',')
    #print(cpFiles)
    #>['results/ViT-B_16_1/orchid_ViT-B_16_checkpoint.bin', 'results/ViT-B_16_1/orchid_ViT-B_16_checkpoint.bin']

    imgSize = args.img_size
    imgSize = imgSize[1:-1]
    imgSize = imgSize.split(',')
    imgSizes = [int(a) for a in imgSize]
    #print(imgSizes)
    #>[384, 384]


    num2cls = dict()
    num2cls = {0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9}    ################################ 2. all class ################################


    
    print(num2cls)

    imgNum = int(len(os.listdir(args.test_dir + '1/')))    # number of all testing data
    #print(imgNum)]

    predictList = [[] for i in range(imgNum)]
    Logits = []

    for i in range(len(modelTypes)):
        args.model_type = modelTypes[i]
        args.checkpoint = cpFiles[i]
        args.img_size = imgSizes[i]

        args, model = setup(args)

        # Test
        pred, groundTruth, lt, imgList = test(args, model)
        #print(np.shape(lt[i]))
        #>(219,)

        for j in range(imgNum):
            predictList[j].append(int(pred[j]))

            if i==0:
                Logits.append(softmax(lt[j]))
            Logits[j]+=softmax(lt[j])


    csvPath1 = '/home/yuhsi/Badminton/src/yolov5/golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LX_LY_case1_HD.csv'    ################################ 3. csv-path ################################
    csvPath2 = '/home/yuhsi/Badminton/src/yolov5/golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD.csv'    ################################ 3. csv-path ################################
    df1 = pd.read_csv(csvPath1)
    df2 = pd.read_csv(csvPath2)

    ################
    # vote ensemble
    ################
    #print(predictList)
    #>[[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [104, 104], [3, 3], [3, 3], [15, 15], [4, 4], [5, 5], [5, 5], [6, 6] ......
    ##[[model1, model2], [model1, model2] ......
    ensemblePred = np.array([np.argmax(np.bincount(a)) for a in predictList])
    #print(ensemblePred)
    #>[  0   0   1   1   2 104   3   3  15   4   5   5   6   6 ......
    print(ensemblePred)

    Hitter = [num2cls[ensemblePred[i]] for i in range(imgNum)]
    df1['BallType'] = Hitter    ################################ 4. attribute ################################
    df1.to_csv('golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LX_LY_case1_HD_balltype_vote.csv', index=False)    ################################ 5. csv-name ################################

    ################
    # mean ensemble
    ################
    ensembleLogits = np.array([np.argmax(a) for a in Logits])
    print(ensembleLogits)

    Hitter = [num2cls[ensembleLogits[i]] for i in range(imgNum)]
    df2['BallType'] = Hitter    ################################ 4. attribute ################################
    df2.to_csv('golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_mean.csv', index=False)    ################################ 5. csv-name ################################


if __name__ == "__main__":
    main()
