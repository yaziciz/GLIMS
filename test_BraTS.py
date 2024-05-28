#Last Modified: 28.05.2024 by Ziya Ata Yazici
#ENSEMBLE TESTING - BraTS 2023 Submision - Post-Processing

import argparse
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
from utils.data_utils_test import get_loader

from monai.inferers import sliding_window_inference

from GLIMS import GLIMS

from monai.transforms import Activations, AsDiscrete

import argparse

import cc3d

import SimpleITK as sitk

#python test_BraTS.py --data_dir /mnt/storage1/dataset/Medical/BraTS2023/Dataset/ValidationData --model_ensemble_1 /mnt/storage1/ziya/BraTS_Models/Archive/HybridEncoder/Log/adjusted_hybrid_file_4_2_poseb_fold2/model_2_new.pt --model_ensemble_2 /mnt/storage1/ziya/BraTS_Models/Archive/HybridEncoder/Log/adjusted_hybrid_file_4_2_poseb_fold4/model_4_new.pt --output_dir /mnt/storage1/ziya/GLIMS/Github

parser = argparse.ArgumentParser(description="GLIMS Brain Tumor Segmentation Pipeline")
parser.add_argument("-f")

parser.add_argument("--data_dir", type=str, help="dataset directory", required=True)
parser.add_argument("--model_ensemble_1", type=str, help="pretrained model name", required=True)
parser.add_argument("--model_ensemble_2", type=str, help="pretrained model name", required=True)
parser.add_argument("--output_dir", type=str, help="Segmentation mask output directory", required=True)

parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--feature_size", default=24, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=2, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

def main():
    args = parser.parse_args()

    args.test_mode = True

    output_directory = os.path.join(args.output_dir, args.exp_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test_loader = get_loader(args) #Get loader of the testing data

    device = torch.device("cuda")

    model = GLIMS(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        )

    model2 = GLIMS(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )

    model_dict = torch.load(args.model_ensemble_1)["state_dict"]
    model_dict2 = torch.load(args.model_ensemble_2)["state_dict"]

    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model2.load_state_dict(model_dict2)
    model2.eval()
    model2.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    model_inferer_test2 = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model2,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():

        post_sigmoid = Activations(sigmoid=True)#output activation
        post_predTC = AsDiscrete(argmax=False, threshold=0.6)
        post_predWT = AsDiscrete(argmax=False, threshold=0.5)
        post_predET = AsDiscrete(argmax=False, threshold=0.6)

        for i, batch in enumerate(test_loader):

            image = batch["image"].cuda()
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = num + ".nii.gz"

            print("Inference on case {}".format(img_name))

            #get logits
            logits = model_inferer_test(image) # 3, 240, 240, 155
            logits2 = model_inferer_test2(image) # 3, 240, 240, 155

            logits = (logits + logits2)/2

            sigmoid = post_sigmoid(logits[0])
            TC = post_predTC(sigmoid[0])
            WT = post_predWT(sigmoid[1])
            ET = post_predET(sigmoid[2])

            val_output_convert = torch.stack([TC, WT, ET])

            seg = val_output_convert
            seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))

            seg_out[seg[1] == 1] = 2 #ED
            seg_out[seg[0] == 1] = 1 #NCR
            seg_out[seg[2] == 1] = 3 #ET
            seg_out = seg_out.cpu().numpy()
           #====================

            #get the ET label
            ET_img = seg_out == 3


            #apply sigmoid pytorch
            npy = post_sigmoid(logits[0]).cpu().numpy()
            
            #cc3d connected component analysis
            cc = cc3d.connected_components(ET_img, connectivity=26)
            for i in np.unique(cc):
                if i == 0:
                    continue

                if(ET_img[cc == i].size < 75):
                    if((npy[-1][cc == i].mean() < 0.9)): #Check ET probability
                        seg_out[cc == i] = 1 #assign ET to NCR

            #======================

            #get the NCR label
            NCR_img = seg_out == 1

            #apply sigmoid pytorch
            npy = post_sigmoid(logits[0]).cpu().numpy()
            
            #cc3d connected component analysis
            cc = cc3d.connected_components(NCR_img, connectivity=26)
            for i in np.unique(cc):
                if i == 0:
                    continue

                if(NCR_img[cc == i].size < 75):
                    if((npy[-3][cc == i].mean() < 0.9)): #Check TC probability
                        seg_out[cc == i] = 2 #assign NCR to ED

            #======================

            ED_img = seg_out == 2

            cc = cc3d.connected_components(ED_img, connectivity=26)
            for i in np.unique(cc):
                if i == 0:
                    continue
                if(ED_img[cc == i].size < 500):
                    if((npy[-2][cc == i].mean() < 0.9)): #Check WT probability
                        seg_out[cc == i] = 0 #ED to background

            #======================

            completeVolume = sitk.GetImageFromArray(seg_out.astype(np.uint8))

            closedcompleteVolume = sitk.BinaryFillhole(completeVolume, fullyConnected= True, foregroundValue=3)
            closedCompleteVolume = sitk.GetArrayFromImage(closedcompleteVolume)

            #count label 1 in completeVolume
            pixCount = np.count_nonzero(closedCompleteVolume != seg_out)
            if(pixCount > 0):
                print("Filling holes for:", img_name, "for", pixCount, "pixels", flush=True)
                seg_out[closedCompleteVolume != seg_out] = 1 #Empty pixels in ET assign them to NCR

            #=========================
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))

if __name__ == "__main__":
    main()
