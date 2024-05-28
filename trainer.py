#Last Modified: 28.05.2024 by Ziya Ata Yazici

import os
import time
import nibabel as nib

import numpy as np
import torch
import torch.nn.parallel
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
import wandb
from utils.utils import AverageMeter

import sys

from monai.data import decollate_batch

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train() #Init the train phase for the model

    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader): #Get the data from the loader in batches

        if torch.isnan(batch_data["image"]).any():
            print("NaN in image")
            continue
        
        start_time = time.time()

        if isinstance(batch_data, list): #Check if the batch is in list or dictionary format.
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        data, target = data.cuda(args.rank), target.cuda(args.rank) #Move them to the GPU

        for param in model.parameters(): #Set the gradient of all parameters to 0
            param.grad = None

        with autocast(enabled=args.amp):
            logits = model(data, "Train")

            #downsample target
            target_uint8 = target.type(torch.uint8)
            target2 = F.interpolate(target_uint8, size=(48, 48, 48), mode='nearest').type(torch.bool)
            target3 = F.interpolate(target_uint8, size=(24, 24, 24), mode='nearest').type(torch.bool)
            target4 = F.interpolate(target_uint8, size=(12, 12, 12), mode='nearest').type(torch.bool)

            loss = loss_func(logits[0], target)
            loss2 = loss_func(logits[1], target2)
            loss3 = loss_func(logits[2], target3)
            loss4 = loss_func(logits[3], target4)

            total_loss = loss + (1/2) * loss2 + (1/4) * loss3 + (1/8) * loss4

            #normalize total loss
            total_loss = total_loss / (1 + (1/2) + (1/4) + (1/8))

        if args.amp: #If AMP is enabled, scale the gradients and update the parameters with the given optimizer
            if torch.isnan(total_loss):
                print("NaN, exiting")
                sys.exit()
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        run_loss.update(total_loss.item(), n=args.batch_size) #Update the AverageMeter
        
        if args.rank == 0: #Print the epoch information
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "train_loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
                flush=True,
            )

        if wandb.run is not None:
            wandb.log({'Iteration_Train_Loss': total_loss.item()}) #Total training loss

    torch.cuda.empty_cache()

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, loss_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()#Take the model to the evaluation phase

    run_metric = []
    for i in range(0, args.out_channels):
        run_metric.append(AverageMeter())

    run_dice = AverageMeter()
    run_loss = AverageMeter()

    output_directory = args.output_dir

    with torch.no_grad(): #Will not perform any gradient operation, thus no_grad.

        for idx, batch in enumerate(loader):
            image = batch["image"].cuda()
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            target = batch["label"].cuda()

            with autocast(enabled=args.amp):
                logits = model_inferer(image) #Use the specified inferer to perform the forward pass.

                #get validation loss
                loss = loss_func(logits, target) #Get the final layer's loss
                run_loss.update(loss)

            val_labels_list = decollate_batch(target)
            val_labels_convert = [val_label_tensor for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]

            #Calculate dice
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc_func.aggregate()[0][0].cpu().numpy()

            for j in range(0, len(acc_func._buffers[0])): #for each image
                for i in range(0, args.out_channels):
                    acc_class = acc_func._buffers[0][j][i].cpu()
                    run_metric[i].update(acc_class.cpu().numpy())

            run_dice.update(acc)

            if(wandb.run is not None and idx == 10):
                #find the index == 1 in one-hot and create single channel
                if not os.path.exists(output_directory): #Sample output (.png) and ground truth, just to show the final validation result on the given index data on local.
                    os.makedirs(output_directory)

                seg = val_output_convert[0]
                seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))

                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 3
                seg_out = seg_out.cpu().numpy()
                nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))

                if(wandb.run is not None):
                    wandb.log({"TC": wandb.Image(seg[0, :, :, 80].cpu().numpy(), caption=f"Dice {run_metric[0].avg}")})
                    wandb.log({"WT":wandb.Image(seg[1, :, :, 80].cpu().numpy(), caption=f"Dice {run_metric[1].avg}")})
                    wandb.log({"ET":wandb.Image(seg[2, :, :, 80].cpu().numpy(), caption=f"Dice {run_metric[2].avg}")})
                    wandb.log({"Original": wandb.Image(image[0, 0, :, :, 80].cpu().numpy(), caption=f"Input Image")})
                    wandb.log({"Original_TC": wandb.Image(val_labels_list[0][0, :, :, 80].cpu().numpy(), caption=f"Target Mask TC")})
                    wandb.log({"Original_WT":wandb.Image(val_labels_list[0][1, :, :, 80].cpu().numpy(), caption=f"Target Mask WT")})
                    wandb.log({"Original_ET":wandb.Image(val_labels_list[0][2, :, :, 80].cpu().numpy(), caption=f"Target Mask ET")})

            torch.cuda.empty_cache()

    return run_metric, run_loss.avg, np.mean(run_dice.avg)


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None, lesion = False):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    path = os.path.join(args.output_dir, filename)

    torch.save(save_dict, path)
    print("Saving checkpoint", path)

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    scaler = None

    if args.amp:
        scaler = GradScaler() #Gradient Scaler to perform smooth backprop

    val_acc_max = 0.0

    for epoch in range(start_epoch, args.max_epochs):

        print(args.rank, time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        
        #Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args)

        print(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "Train_Loss: {:.4f}".format(train_loss),
            "Time {:.2f}s".format(time.time() - epoch_time),
            "Learning_Rate:{}".format(optimizer.param_groups[0]["lr"]),
        )

        if wandb.run is not None:
            wandb.log({'Train_Loss': train_loss}) #Add the avg. train loss of each epoch
            wandb.log({'Learning_Rate': optimizer.param_groups[0]["lr"]}) #Add the learning rate of each epoch

        if (epoch + 1) % args.val_every == 0: # Perform validation check for a given frequency (args.val_every), including the initial one

            epoch_time = time.time()

            val_acc, val_loss, val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if wandb.run is not None:
                wandb.log({'Dice_TC': val_acc[0].avg})
                wandb.log({'Dice_WT': val_acc[1].avg})
                wandb.log({'Dice_ET': val_acc[2].avg})
                wandb.log({'Val_Loss': val_loss}) ##Add the learning rate of each epoch
                wandb.log({'Mean_Val_Dice': val_avg_acc}) ##Add the learning rate of each epoch

            if args.rank == 0:
                Dice_TC = val_acc[0].avg
                Dice_WT = val_acc[1].avg
                Dice_ET = val_acc[2].avg

                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice_TC:",
                    Dice_TC,
                    ", Dice_WT:",
                    Dice_WT,
                    ", Dice_ET:",
                    Dice_ET,
                    "Mean_Dice:",
                    val_avg_acc,
                    flush=True,
                    )
                
                #New best model was achieved
                if val_avg_acc > val_acc_max:
                    print("New Best for Legacy Dice! ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    
                    if args.rank == 0 and args.output_dir is not None and args.save_checkpoint:
                        #Save the model checkpoint
                        print("Saving the new best model!")
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler)

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    if args.output_dir is not None and args.save_checkpoint:
        print("Saving the final model!")
        save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

    return val_acc_max