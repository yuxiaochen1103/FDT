import torch
from .transforms import build_transformer
from .imagenet_dataloader import build_common_augmentation
import os
from .datasets.clip_dataset import ClipDataset


def build_clip_dataloader(data_type, cfg):
    #check configure
    """
    arguments:
        - data_type: 'train', 'test', 'val'
        - cfg_dataset: configurations of dataset
    """
    print(cfg.keys())
    cfg_dataset = cfg.data
    #get local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    assert data_type in cfg_dataset

    #---build transformations
    if isinstance(cfg_dataset[data_type]['transforms'], list):
        transforms = build_transformer(cfgs=cfg_dataset[data_type]['transforms'])
    else:
        transforms = build_common_augmentation(
            cfg_dataset[data_type]['transforms']['type'])
            
    # # build evaluator
    # evaluator = None
    # if data_type == 'test' and cfg_dataset[data_type].get('evaluator', None):
    #     evaluator = build_evaluator(cfg_dataset[data_type]['evaluator'])

    dataset = ClipDataset(
                 config=cfg,
                 data_path=cfg_dataset[data_type].data_path,
                 transform=transforms,
                 rank=int(os.environ.get('RANK') or 0),
                 world_size=int(os.environ.get('WORLD_SIZE') or 1),
                 shuffle=True,
                 repeat=True, # repeat for multiple epochs
                 preprocess_mode='torchvision',
                 )

    # if config.get('train.do_prefetch', False):
    #     train_loader = DataLoaderX(local_rank=local_rank,
    #                                 dataset=train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
    #                                 num_workers=config.train.num_workers,
    #                                 pin_memory=True,
    #                                 drop_last=True,
    #                                 collate_fn=train_dataset.collect_fn)
    # else:
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg_dataset.batch_size,
                                               num_workers=cfg_dataset.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=dataset.collect_fn)


    return {'type': data_type, 'loader': loader}
