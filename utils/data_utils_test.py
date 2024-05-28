import json
import math
import os

import numpy as np
import torch

from monai import data, transforms


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(basedir):

    #store the path of subfolders in basedir to a list
    val = []
    for root, a, files in os.walk(basedir):
        a = sorted(a)
        for i in range(len(a)):
            temp = a[i].split('/')[-1]
            d = []
            d.append(os.path.join(root, temp, temp + "-t2f.nii.gz"))
            d.append(os.path.join(root,  temp, temp + "-t1c.nii.gz"))
            d.append(os.path.join(root,  temp, temp + "-t1n.nii.gz"))
            d.append(os.path.join(root,  temp, temp + "-t2w.nii.gz"))
            #create dictionary key with "image"
            d = {"image": d}
            val.append(d)
        break
    return val


def get_loader(args):
    data_dir = args.data_dir
    validation_files = datafold_read(basedir=data_dir)

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    if args.test_mode: #Just use the validation data and create a loader.

        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader

    return loader