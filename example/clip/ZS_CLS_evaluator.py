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
import torch.nn.functional as F
from prototype.solver.base_solver import BaseSolver
from prototype.utils.torch_ddp_dist import get_rank, get_world_size, get_local_rank, set_random_seed, init_ddp, \
    convert_to_ddp_model
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, load_state_model, parse_pt_args, \
    parse_config
from prototype.model import model_entry

from prototype.data.img_cls_dataloader import build_imagenet_test_dataloader


# ---------support imagenet zero-shot testing only
class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = link.get_rank()
        ctx.world_size = link.get_world_size()

        #         y = tensor.new(ctx.world_size, *tensor.size())

        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]

        link.allgather(y, tensor)  # call pytorch all togherer

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
        # self.config_file = args.config
        self.args = args
        self.prototype_info = EasyDict()  # a dict

        self.config = parse_config(args.config)

        self.setup_env()
        # import ipdb
        # ipdb.set_trace()
        self.build_model()
        self.build_data()

    def setup_env(self):

        # set random seed
        set_random_seed()
        self.dist = EasyDict()
        # self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()

        # ----get local rank, global rank and world size
        self.dist.local_rank, self.dist.rank, self.dist.world_size = get_local_rank(), get_rank(), get_world_size()
        self.prototype_info.world_size = self.dist.world_size

        # -----set output directories
        self.path = EasyDict()
        # self.path.root_path = os.path.dirname(self.config_file)
        self.path.root_path = self.args.output_path

        # make local dir
        makedir(self.path.root_path)

        # create logger
        self.path.log_path = os.path.join(self.path.root_path, 'zs-imagenet.txt')
        create_logger(self.path.log_path)  # local
        self.logger = get_logger(__name__)

        # --------------
        self.logger.critical(f'config: {pprint.pformat(self.config)}')

        # load checkpoint:
        ckpt_pth = self.args.ckpt_path

        self.state = torch.load(ckpt_pth)  
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

        from prototype.data.imagenet_dataloader import build_common_augmentation
        self.preprocess = build_common_augmentation('ONECROP')

        self.val_data = build_imagenet_test_dataloader(base_fold=self.args.data_fold, split='val', transforms=self.preprocess,
                                                       global_rank=self.dist.rank,
                                                       world_size=self.dist.world_size)  # to do

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def get_logits(self, dense_feat_1, selected_feat_2):
        i, j, k = dense_feat_1.shape
        l, m, k = selected_feat_2.shape
        dense_feat_1 = dense_feat_1.reshape(-1, k)
        selected_feat_2 = selected_feat_2.reshape(-1, k)
        final_logits_1 = dense_feat_1 @ selected_feat_2.t()
        final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3)

        return final_logits_1.max(dim=-1)[0].mean(dim=-1)

    @torch.no_grad()
    def evaluate(self, val_data):
        self.model.eval()


        # -----------extract feature embeddings of label prompt
        label_text, label_text_ensemble_matrix = val_data['loader'].dataset.get_label_texts()
        label_num = label_text_ensemble_matrix.shape[1]
        prompts_num = len(label_text) // label_num
        self.logger.info('Use {} prompts'.format(prompts_num))

        v_dim = 512

        txt_ft_list = torch.zeros(label_num, v_dim).cuda()


        self.logger.info('extracting label embeddings.....')
        for i in range(label_num):
            # for each category extract its label
            txt_ft = self.model.module.encode_text(
                label_text[i * prompts_num:(i + 1) * prompts_num])
            # 1/0
            # l2 normalize features
            txt_ft = txt_ft / (txt_ft.norm(dim=-1, keepdim=True) + 1e-10)

            # mean pool for features
            txt_ft = txt_ft.mean(dim=0)
            txt_ft /= txt_ft.norm()

            txt_ft_list[i] = txt_ft

        if self.dist.rank == 0:
            self.logger.info(f'txt_ft_list.shape:{txt_ft_list.shape}')

        correct_num = 0
        data_num = 0
        # ----------extract image features

        all_labels = []
        all_preds = []
        for batch_idx, batch in enumerate(val_data['loader']):

            images = batch['image']
            label = batch['label'].cuda()
            images = images.cuda()

            # extrac text features
            img_ft = self.model.module.encode_image(images)

            # l2 norm features and score
            img_ft = img_ft / (img_ft.norm(dim=-1, keepdim=True) + 1e-10)

            # get temp
            logit_scale = self.model.module.logit_scale.exp()
            logit_scale.data = torch.clamp(logit_scale.data, max=100)

            # -----cal cos_sim
            bs = img_ft.shape[0]

            # clip view
            logits = img_ft @ txt_ft_list.t()
            assert logits.shape == (bs, label_num)
            logits = logits * logit_scale
            logits = F.softmax(logits, dim=-1)


            # feature view

            # -------get prd
            # score view
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)


            preds = self.all_gather(preds)

            label = self.all_gather(label)

            if self.dist.rank == 0:
                correct_num += (preds == label).sum().item()
                data_num += label.shape[0]

                all_labels += label.data.cpu().numpy().tolist()
                all_preds += preds.data.cpu().numpy().tolist()

            link.barrier()

        if self.dist.rank == 0:
            self.logger.critical('top 1 acc: {}%'.format(correct_num / data_num * 100))


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
