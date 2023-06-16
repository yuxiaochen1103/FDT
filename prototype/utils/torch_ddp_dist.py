import os
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))

def get_rank():
    return int(os.environ.get('RANK', 0))

def is_main_process():
    return get_rank() == 0

def set_random_seed():
    seed = 0
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_ddp():
    local_rank = get_local_rank()
    world_size = get_world_size()
    rank = get_rank()

    print('init_ddp, local_rank:{} , rank:{}, world_size:{}'.format(local_rank, rank, world_size))
    
    torch.cuda.set_device(local_rank)

    # if self.accelerator_syncbn:
    #     #     model = self.configure_sync_batchnorm(model)

    #read master add from os.enbiro
    master_address = os.environ.get('MASTER_ADDR', "0.0.0.0")
    master_port = int(os.environ.get('MASTER_PORT', random.randint(6000, 60000)))
    print(master_address, master_port)
    distributed.init_process_group(
        backend='nccl',
        init_method='tcp://{}:{}'.format(master_address, master_port),
        world_size=world_size,
        rank=rank,
        group_name='mtorch')

def convert_to_ddp_model(model, local_rank, find_unused_parameters=True):

    def broadcast(model, src=0):
        """
        将model的参数做broadcast
        """
        for v in model.state_dict().values():
            distributed.broadcast(v, src)


    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)
    broadcast(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)

    return model


# def set_up(local_rank: int, world_size: int, rank: int, seed:int):
#     """
#     初始化 DDPAccelerator
#     """
#     torch.backends.cudnn.benchmark = False
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
#     torch.cuda.set_device(local_rank)
#
#     # if self.accelerator_syncbn:
#     #     model = self.configure_sync_batchnorm(model)
#
#     model = model.cuda()
#     if not torch.distributed.is_initialized():
#
#
#
#     self.broadcast(model)
#     model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
#
#     return model, optimizer, lr_scheduler


