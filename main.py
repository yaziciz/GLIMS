#Last Modified: 27.05.2024 by Ziya Ata Yazici

import argparse
import os
from functools import partial
import monai
import wandb

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, PolyLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from GLIMS import GLIMS
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

import logging
logging.disable(logging.WARNING)

#python main.py --output_dir /mnt/storage1/ziya/GLIMS/Github/outputs/ --data_dir /mnt/storage1/dataset/Medical/BraTS2023/Dataset --json_list /mnt/storage1/dataset/Medical/BraTS2023/Dataset/brats23_folds.json 

parser = argparse.ArgumentParser(description="GLIMS Brain Tumor Segmentation Pipeline")

parser.add_argument("--data_dir", type=str, help="dataset directory", required=True)
parser.add_argument("--json_list", type=str, help="dataset json file", required=True)

parser.add_argument("--fold", default=0, type=int, help="data fold")

parser.add_argument("--pretrained_dir", default=None, type=str, help="Pretrained model directory")
parser.add_argument("--output_dir", default="/", type=str, help="output directory")

parser.add_argument("--test_mode", default=False, type=bool, help="test mode")

parser.add_argument("--save_checkpoint", default = True, help="save checkpoint during training")  
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs") #500

parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--batch_size_val", default=4, type=int, help="number of batch size for validation") #Does not have a backprop path, so can be larger.
parser.add_argument("--sw_batch_size", default=8, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=0.001, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")

parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")

parser.add_argument("--amp", default=False, help="use amp for training") #AMP performs training faster, but high possibility to receive NaNs.

parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--perform_test", default=False, type=bool, help="testing dataset check")
parser.add_argument("--distributed", default=False, help="start distributed training")

parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

parser.add_argument("--workers", default=8, type=int, help="number of workers") #8

parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")

parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
parser.add_argument("--infer_overlap", default=0.8, type=float, help="sliding window inference overlap")

parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=100, type=int, help="number of warmup epochs")

parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

parser.add_argument("--seed", default=25, help="the random seed to produce deterministic results")

parser.add_argument("--wandb_enable", default=False, help="enable wandb logging")
parser.add_argument("--wandb_project_name", default="GLIMS_Project", help="the name that will be given to the WandB project")

parser.add_argument("--feature_size", default=24, type=int, help="feature size")

parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

#torch.autograd.set_detect_anomaly(True)

def main():
    args = parser.parse_args() #Parse the inputs
    
    if(args.wandb_enable):
        
        #🐝 initialize a wandb run
        wandb.init(
            project=args.wandb_project_name,
            config=args,
            reinit=True,
            name="GLIMS",
        )

    #For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    monai.utils.misc.set_determinism(seed=args.seed)

    main_worker(args=args)

def main_worker(args):

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    torch.cuda.set_device(args.gpu)

    loader = get_loader(args) #Loader of the dataset (both training and validation)

    print(args.rank, " gpu", args.gpu)

    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    #Create the model.
    model = GLIMS(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )
    
    dice_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=0.0, include_background=True)

    post_sigmoid = Activations(sigmoid=True)#output activation

    post_pred = AsDiscrete(argmax=False, threshold=0.5) 

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, ignore_empty=False) #Mean dice loss throught the batch.

    #The inferer model that will perform the validation, "sliding_window_inference"
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    #----------------------------------------- TRAINING ----------------------------------------------------
    best_acc = 0
    start_epoch = 0

    #If there is a checkpoint, load it to the model.
    if args.use_checkpoint is True:
        model_dict = torch.load(pretrained_dir)["state_dict"]
        model.load_state_dict(model_dict)
        print("Pretrained model loaded from: ", pretrained_pth)

    model.cuda(args.gpu)

    #Parallel training on multiple GPUs if available.
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    #Optimizers
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    #LR Schedulers
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.pretrained_dir is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "poly":
        scheduler = PolyLRScheduler(optimizer, initial_lr=args.optim_lr, max_steps=args.max_epochs)
        if args.pretrained_dir is not None:
            scheduler.step(current_step=start_epoch)
    elif args.lrschedule == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True)
    else:
        scheduler = None

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch, #will be 0 if no checkpoint was imported.
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
