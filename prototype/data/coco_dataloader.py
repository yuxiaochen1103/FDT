from .datasets.coco_dataset import COCO_Dataset
from .transforms import build_transformer
import os
import torch
from .imagenet_dataloader import build_common_augmentation


def build_coco_dataloader(data_fold, global_rank, world_size, is_train):

    #---get transformation
    transforms = build_common_augmentation('ONECROP')

    # build sampler
    #to do.... check config
    dataset = COCO_Dataset(
        base_fold=data_fold,
        is_train=is_train,
        transform=transforms)

    sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=False
        )

    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=512,
                                            num_workers=8,
                                            pin_memory=True,
                                            drop_last=False,
                                            sampler=sampler,
                                            collate_fn=dataset.collect_fn)

    return {'loader': loader}