#Last Modified: 28.05.2024 by Ziya Ata Yazici

import argparse
import os
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from torch.cuda.amp import autocast

from GLIMS import GLIMS
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

from utils.utils import AverageMeter
import nibabel as nib
import pandas as pd

#python post_validation.py --json_list /mnt/storage1/dataset/Medical/BraTS2023/Dataset/brats23_folds.json --pretrained_dir /mnt/storage1/ziya/BraTS_Models/Archive/HybridEncoder/Log/adjusted_hybrid_file_4_2_poseb_fold2/model_2_new.pt --fold 2 --output_dir /mnt/storage1/ziya/GLIMS/Github --data_dir /mnt/storage1/dataset/Medical/BraTS2023/Dataset

import logging
logging.disable(logging.WARNING)

parser = argparse.ArgumentParser(description="GLIMS Brain Tumor Segmentation Pipeline")

parser.add_argument("--data_dir", type=str, help="dataset directory", required=True)
parser.add_argument("--json_list", type=str, help="dataset json file", required=True)
parser.add_argument("--pretrained_dir", type=str, help="pretrained model name", required=True)

parser.add_argument("--output_dir", default="/", type=str, help="Saved model directory")

parser.add_argument("--fold", default=2, type=int, help="data fold")

parser.add_argument("--test_mode", default=True, type=bool, help="test mode")

parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--batch_size_val", default=1, type=int, help="number of batch size for validation") #Does not have a backprop path, so can be larger.
parser.add_argument("--sw_batch_size", default=8, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=0.001, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")

parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")

parser.add_argument("--amp", default=False, help="use amp for training") #AMP performs training faster, but high possibility to receive NaNs.

parser.add_argument("--val_every", default=1, type=int, help="validation frequency") #Val every 50
parser.add_argument("--perform_test", default=False, type=bool, help="testing dataset check") #Save every 50
parser.add_argument("--distributed", default=False, help="start distributed training")

parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

parser.add_argument("--workers", default=8, type=int, help="number of workers") #8

parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")

parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
parser.add_argument("--infer_overlap", default=0.8, type=float, help="sliding window inference overlap")

parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

parser.add_argument("--seed", default=25, help="the random seed to produce deterministic results")

parser.add_argument("--feature_size", default=24, type=int, help="feature size")

parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

def main():

    args = parser.parse_args()
    device = torch.device("cuda")

    model = GLIMS(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size
    )

    model_dict = torch.load(args.pretrained_dir)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()#Take the model to the evaluation phase
    model.to(device)

    checkpoint = torch.load(args.pretrained_dir, map_location="cuda")
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        new_state_dict[k.replace("backbone.", "")] = v
    model.load_state_dict(new_state_dict, strict=False)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    loss_func = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    post_sigmoid = Activations(sigmoid=True)#output activation
    post_pred = AsDiscrete(argmax=False, threshold=0.5) 
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True, ignore_empty=False) #Mean dice loss throught the batch.
    hd_func = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction=MetricReduction.MEAN, get_not_nans=True)

    run_metric_dice_ET = AverageMeter()
    run_metric_dice_WT = AverageMeter()
    run_metric_dice_TC = AverageMeter()
    run_metric_hd_ET = AverageMeter()
    run_metric_hd_WT = AverageMeter()
    run_metric_hd_TC = AverageMeter()
    run_loss = AverageMeter()

    df_array = []

    loader = get_loader(args) # Get the validation loader

    args.output_folder = os.path.join(args.output_dir, "fold" + str(args.fold))
    #if no directory exists, create one
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with torch.no_grad(): #Will not perform any gradient operation, thus no_grad.

        for idx, batch in tqdm(enumerate(loader)):
            image = batch["image"].cuda()
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
            img_name = "BraTS2023_" + num + ".nii.gz"
            target = batch["label"].cuda()

            with autocast(enabled=args.amp):
                logits = model_inferer(image) #Use the specified inferer to perform the forward pass.

                #get validation loss
                loss = loss_func(logits, target) #Get the final layer's loss
                run_loss.update(loss.item())

            #Calculate dice
            acc_func.reset()

            val_labels_list = target[0]
            val_outputs_list = logits[0]
            val_output_convert = post_pred(post_sigmoid(val_outputs_list))
            
            acc_func(y_pred=val_output_convert.unsqueeze(0), y=val_labels_list.unsqueeze(0))

            TC_dice = acc_func._buffers[0][0][0].cpu().numpy()
            WT_dice = acc_func._buffers[0][0][1].cpu().numpy()
            ET_dice = acc_func._buffers[0][0][2].cpu().numpy()

            run_metric_dice_ET.update(ET_dice)
            run_metric_dice_WT.update(WT_dice)
            run_metric_dice_TC.update(TC_dice)

            #Calculate HD
            hd_func.reset()
            hd_func(y_pred=val_output_convert.unsqueeze(0), y=val_labels_list.unsqueeze(0))

            TC_hd = hd_func._buffers[0][0][0].cpu().numpy()
            WT_hd = hd_func._buffers[0][0][1].cpu().numpy()
            ET_hd = hd_func._buffers[0][0][2].cpu().numpy()
            run_metric_hd_ET.update(ET_hd)
            run_metric_hd_WT.update(WT_hd)
            run_metric_hd_TC.update(TC_hd)

            seg = val_output_convert
            seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))

            seg_out[seg[1] == 1] = 2 #Edema
            seg_out[seg[0] == 1] = 1 #TC
            seg_out[seg[2] == 1] = 3 #ET
            seg_out = seg_out.cpu().numpy()

            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(args.output_folder, img_name))

            #sstore each score as a df line, and save all to a single file
            df = pd.DataFrame({"TC_dice": TC_dice, "WT_dice": WT_dice, "ET_dice": ET_dice, "TC_hd": TC_hd, "WT_hd": WT_hd, "ET_hd": ET_hd}, index=[num])
            df_array.append(df)

    df = pd.concat(df_array)
    df.to_excel(os.path.join(args.output_folder, "scores.xlsx"))

    print("Done")
    print("Mean Dice: ", run_metric_dice_ET.avg, run_metric_dice_WT.avg, run_metric_dice_TC.avg)
    print("Mean HD: ", run_metric_hd_ET.avg, run_metric_hd_WT.avg, run_metric_hd_TC.avg)

if __name__ == "__main__":
    main()
