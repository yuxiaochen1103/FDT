import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


import argparse
from easydict import EasyDict
import pprint
import torch
import prototype.linklink as link
from prototype.solver.base_solver import BaseSolver
from prototype.utils.torch_ddp_dist import get_rank, get_world_size, get_local_rank, set_random_seed, init_ddp, convert_to_ddp_model
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, load_state_model, parse_config
from prototype.model import model_entry
import numpy as np
from prototype.data.coco_dataloader import build_coco_dataloader
import pickle

#---------support imagenet zero-shot testing only

class AllGather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = link.get_rank()
        ctx.world_size = link.get_world_size()

#         y = tensor.new(ctx.world_size, *tensor.size())
        
        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        
        link.allgather(y, tensor) #call pytorch all togherer

        y = torch.cat(y, 0).view(-1, *tensor.size())
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        link.allreduce(in_grad)
        # split
        return in_grad[ctx.rank]


class ClsSolver(BaseSolver):
    def __init__(self, args):
        #self.config_file = args.config
        self.args = args
        self.prototype_info = EasyDict() #a dict
        #config parameters

        self.config = parse_config(args.config)

        self.setup_env()
        self.build_model()
        self.build_data()

    def setup_env(self):
        #set random seed
        set_random_seed()

        #set distributed setting
        self.dist = EasyDict()

        #----get local rank, global rank and world size
        self.dist.local_rank, self.dist.rank, self.dist.world_size = get_local_rank(), get_rank(), get_world_size()
        self.prototype_info.world_size = self.dist.world_size

        #-----set output directories
        self.path = EasyDict()

        self.path.root_path = self.args.output_path

        #make local dir
        makedir(self.path.root_path)

        # create logger
        self.path.log_path = os.path.join(self.path.root_path, 'zs-coco.txt')
        create_logger(self.path.log_path) #local
        self.logger = get_logger(__name__)

        #--------------
        self.logger.critical(f'config: {pprint.pformat(self.config)}')

        #load checkpoint:
        ckpt_pth = self.args.ckpt_path

        self.state = torch.load(ckpt_pth, map_location='cpu')
        self.logger.info(f"load ckpt from {ckpt_pth}")
        self.logger.info(f"self.state keys {self.state.keys()}")

        # others
        torch.backends.cudnn.benchmark = True


    def build_model(self):
        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)

        self.model = convert_to_ddp_model(self.model, self.dist.local_rank)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_data(self):
        self.logger.info('loading test dataset')
        self.val_data = build_coco_dataloader(data_fold=self.args.data_fold, global_rank=self.dist.rank,
                                              world_size=self.dist.world_size, is_train=False) #to do

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def cal_i2t_metric(self, sim_matrix, index_list):
        #return r1, r5, and r10 for image to text retrieval
        assert len(sim_matrix) == len(index_list)
        k_list = [1, 5, 10]
        max_k = max(k_list)
        cor_num = np.array([0 for _ in k_list])
        img_num = len(sim_matrix)

        for i in range(img_num):
            #get idx of top max_k
            score, prd = torch.sort(sim_matrix[i], descending=True)[:max_k]
            prd = prd[:max_k]

            imgidx = index_list[i]
            gt_idx_begin = imgidx * 5
            gt_idx_end = gt_idx_begin + 5

            for kidx, k in enumerate(k_list):
                k_prd = prd[:k]
                is_correct = ((k_prd >= gt_idx_begin) * (k_prd < gt_idx_end)).sum() > 0
                if is_correct:
                    cor_num[kidx] += 1
        
        acc = cor_num / img_num * 100
        acc = np.around(acc, 1)
        self.logger.critical('\n')
        self.logger.critical('-'*30)
        self.logger.critical('Image to Text:')
        for kidx, k in enumerate(k_list):
            self.logger.critical('R@{}: {}%'.format(k, acc[kidx]))
        self.logger.critical('sum: {}'.format(acc.sum()))
        self.logger.critical('-'*30)

        return acc
    
    def cal_t2i_metric(self, sim_matrix, index_list):
        #return r1, r5, and r10 for text to image retrieval
        sim_matrix = torch.transpose(sim_matrix, 0, 1)
        index_list = np.array(index_list)
        assert sim_matrix.shape[1] == len(index_list)
        k_list = [1, 5, 10]
        max_k = max(k_list)
        cor_num = np.array([0 for _ in k_list])
        cap_num = len(sim_matrix)

        for cap_idx in range(cap_num):
            score, prd = torch.sort(sim_matrix[cap_idx], descending=True)[:max_k]
            prd = prd[:max_k]
            gt_idx = cap_idx // 5
            for kidx, k in enumerate(k_list):
                k_prd = prd[:k].numpy().tolist()
                retr_imgidx_list = np.array(index_list[k_prd])
                is_correct = (retr_imgidx_list == gt_idx).sum() > 0
                if is_correct :
                    cor_num[kidx] += 1
        
        acc = cor_num / cap_num * 100
        acc = np.around(acc, 1)
        
        self.logger.critical('\n')
        self.logger.critical('-'*30)
        self.logger.critical('Text to Image:')
        for kidx, k in enumerate(k_list):
            self.logger.critical('R@{}: {}%'.format(k, acc[kidx]))
        self.logger.critical('sum: {}'.format(acc.sum()))
        self.logger.critical('-'*30)

        return acc
    
    @torch.no_grad()
    def evaluate(self, val_data):


        self.model.eval()

        curr_step = self.state['last_iter']
        sd_T = self.config.t_decay.org_t #orginal temperature
        sd_T_decay_w = self.config.t_decay.sd_T_decay_w #temperature decay weight
        sd_T_decay_iter = self.config.t_decay.sd_T_decay_iter #temperature decay iter
        sd_T_min = self.config.t_decay.sd_T_min#minimum temperature

        temperature = sd_T * (sd_T_decay_w ** (curr_step / sd_T_decay_iter))
        temperature = max(temperature, sd_T_min)

        self.model.module.img_query_model.temperature = temperature
        self.model.module.txt_query_model.temperature = temperature
        
        self.logger.critical('step: {} temp: {}'.format(curr_step, self.model.module.img_query_model.temperature)) 

        ##------extact captions embeddings
        all_cap = val_data['loader'].dataset.allcap
        self.logger.info('extact captions embeddings......')
        bz = 512
        cap_num = len(all_cap)
        begin_idx = 0
        cap_emb_list = []

        while begin_idx < cap_num:
            _, cap_emb, _ = self.model.module.extract_txt_sd_ft(all_cap[begin_idx : begin_idx+bz])
            cap_emb /= cap_emb.norm(dim=-1, keepdim=True)
            cap_emb_list.append(cap_emb)
            begin_idx += bz

        cap_emb_list = torch.cat(cap_emb_list, dim=0)
        if self.dist.rank == 0:
            self.logger.info(f'cap_emb_list.shape:{cap_emb_list.shape}')
        assert len(cap_emb_list) == cap_num

        #------initl similarity matrix
        if self.dist.rank == 0:

            img_num = len(val_data['loader'].dataset.imgname_list)
            txt_num = len(all_cap)
            sim_matrix = torch.zeros(img_num, txt_num) #row: image-to-text sim, column, text_to_img sim
            sim_idx = 0
            index_list = []


        #calculate image and text similarity
        for batch_idx, batch in enumerate(val_data['loader']):
            if self.dist.rank == 0:
                self.logger.info('calculate image and text similarity [{} of {}]'.format(batch_idx+1, len(val_data['loader'])))
            input = batch['images']
            index = batch['indexs'].to(cap_emb_list)
            input = input.cuda()

            #get image features\
            _, image_preds, _ = self.model.module.extract_img_sd_ft(input)

            #l2 norm
            image_preds = image_preds / \
                (image_preds.norm(dim=-1, keepdim=True))

            #cosine simarity
            logits = image_preds @ cap_emb_list.t() #image to text similarity
            # compute prediction
            # all_gather 
            logits = self.all_gather(logits)
            indexs = self.all_gather(index) #gt index

            assert logits.shape[0] == input.shape[0] * self.dist.world_size == indexs.shape[0]

            if self.dist.rank == 0:
                #record sim matrix
                bz = logits.shape[0]
                sim_matrix[sim_idx:sim_idx+bz] = logits.data.cpu()
                sim_idx += bz
                index_list.extend(indexs.data.cpu().numpy().tolist())
            link.barrier()

        if self.dist.rank == 0:
            self.logger.critical('coco: temperature: {}'.format(self.model.module.img_query_model.temperature ))
            assert sim_idx == img_num == len(index_list)
            t2i_acc = self.cal_t2i_metric(sim_matrix, index_list)
            i2t_acc = self.cal_i2t_metric(sim_matrix, index_list)
            self.logger.critical('rsum: {}'.format(t2i_acc.sum() + i2t_acc.sum()))

def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--ckpt_path', required=True, type=str)
    parser.add_argument('--data_fold', required=True, type=str)

    args = parser.parse_args()

    # set up pytorch ddp
    init_ddp()

    # build solver
    solver = ClsSolver(args)
    # evaluate or train
    
    solver.evaluate(solver.val_data)

if __name__ == '__main__':
    main()
