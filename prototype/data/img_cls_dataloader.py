from .datasets.imagenet import ImgnetDataset
import torch

#to do---
def build_imagenet_test_dataloader(base_fold, split, transforms, global_rank, world_size):

    base_fold = base_fold + '/ImageNet-1K'
    dataset = ImgnetDataset(
        split='val',
        root_dir=base_fold,
        transform=transforms
    )


    sampler= torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=False)

    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=64,
                                            sampler=sampler,
                                            num_workers=8,
                                            pin_memory=True,
                                            drop_last=False)


    return {'loader': loader}