import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import prototype.linklink as link
import json


from prototype.solver.base_solver import BaseSolver
from prototype.utils.torch_ddp_dist import get_rank, get_world_size, get_local_rank, set_random_seed, init_ddp, convert_to_ddp_model
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, \
    param_group_all, AverageMeter, accuracy, load_state_optimizer,parse_config
from prototype.utils.ema import EMA
from prototype.model import model_entry
from prototype.optimizer import optim_entry
from prototype.lr_scheduler import scheduler_entry
from prototype.loss_functions import ClipInfoCELoss
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_

from prototype.data.datasets.clip_dataset_wsd import get_wds_dataset
from prototype.data.img_cls_dataloader import build_imagenet_test_dataloader


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


class EMA_logit_scale():
    def __init__(self, param, threshold):
        self.param = param
        self.buffer = 3.125
        self.momentum = 0.9
        self.threshold = threshold
        self.clip_number = 0

    def update(self):
        self.buffer = self.momentum*self.buffer + \
            (1-self.momentum)*self.param.data

    def clamp(self):
        if (self.param-self.buffer) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer+self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        elif (self.buffer-self.param) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer-self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        # self.param.data = torch.as_tensor(
        #     3.125, dtype=self.param.dtype, device=self.param.device)


class ClsSolver(BaseSolver):
    def __init__(self, args):
        self.args = args
        self.config_file = args.config
        self.prototype_info = EasyDict() #a dict
        self.config = parse_config(self.config_file)
        self.config.data.batch_size = args.batch_size

        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        #set random seed
        set_random_seed()
        self.dist = EasyDict()

        #----get local rank, global rank and world size
        self.dist.local_rank, self.dist.rank, self.dist.world_size = get_local_rank(), get_rank(), get_world_size()
        self.prototype_info.world_size = self.dist.world_size

        #-----set output directories
        self.path = EasyDict()
        self.path.output_path = self.args.output_path

        #local
        self.path.save_path = os.path.join(self.path.output_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.output_path, 'events')
        self.path.result_path = os.path.join(self.path.output_path, 'results') #save location as the config_file


        #make local dir
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)

        # create logger
        self.path.log_path = os.path.join(self.path.output_path, 'log.txt')
        create_logger(self.path.log_path) #local
        self.logger = get_logger(__name__)

        # create tb_logger for the main process
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)

            config_pth = self.path.output_path + '/config.json'
            with open(config_pth, 'w') as file:
                json.dump(self.config, file)

            self.logger.critical('saved configs to {}'.format(self.path.output_path + '/config.json'))

        #--------------
        self.logger.critical(f'config: {pprint.pformat(self.config)}')
        #----------------------

        self.state = {}
        self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_model(self):


        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)


        self.model = convert_to_ddp_model(self.model, self.dist.local_rank)



    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # set non weight-decay parameter
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}
            pconfig['ln_w'] = {'weight_decay': 0.0}
            pconfig['ln_b'] = {'weight_decay': 0.0}


        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        #----optimizer
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])


    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        test_config = {}
        self.config.data.last_iter = self.state['last_iter']
        test_config['last_iter'] = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
            test_config['max_iter'] = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch
            test_config['max_epoch'] = self.config.lr_scheduler.kwargs.max_epoch

        self.train_data = get_wds_dataset(self.config.data.train, world_size=get_world_size())

        from prototype.data.imagenet_dataloader import build_common_augmentation
        preprocess = build_common_augmentation('ONECROP')
        self.val_data = build_imagenet_test_dataloader(base_fold=self.config.data.test.data_fold, split='val', transforms=preprocess,
                                                       global_rank=self.dist.rank,
                                                       world_size=self.dist.world_size)


    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        self.topk = 5
        self.criterion = ClipInfoCELoss()


    def train(self):
        self.pre_train() #set up for setting

        #set training steps by epcoh
        dataloader = self.train_data.dataloader
        each_epoch_step = dataloader.num_batches
        epoch = self.config.data.train.epoch
        total_step = epoch * each_epoch_step

        if self.dist.rank == '0':
            self.logger.critical(
                'total_step: {}'.format(total_step))

        #step
        start_step = self.state['last_iter']
        curr_step = start_step


        end = time.time()
        last_logit_scale = 0
        logit_scale = EMA_logit_scale(self.model.module.logit_scale,
                          self.config.grad_clip.value)
        train_loss = 1e9

        for epoch_id in range(epoch):
            self.train_data.set_epoch(epoch_id) #set epoch for dataloader
            for i, (image, text) in enumerate(dataloader):

                #learing rate scheduler, by steps
                curr_step += 1
                self.lr_scheduler.step(curr_step) #learning rate is calculated based on step
                current_lr = self.lr_scheduler.get_lr()[0]

                # measure data loading time
                self.meters.data_time.update(time.time() - end)

                image = image.cuda()

                # forward
                logits_per_image, logits_per_text = self.model(image, text) #did it fuse multi-gpu>

                # loss
                loss, target = self.criterion(logits_per_image, logits_per_text)
                loss /= self.dist.world_size

                # measure accuracy and record loss
                prec1, prec5 = accuracy(
                    logits_per_image, target, topk=(1, self.topk))

                reduced_loss = loss.clone()
                reduced_prec1 = prec1.clone() / self.dist.world_size
                reduced_prec5 = prec5.clone() / self.dist.world_size

                #update meter
                self.meters.losses.reduce_update(reduced_loss)
                self.meters.top1.reduce_update(reduced_prec1)
                self.meters.top5.reduce_update(reduced_prec5)

                #
                if curr_step % 50 == 0:
                    self.logger.info(f'Epoch[{epoch_id+1}] Iter[{curr_step}]: self.meters.losses.avg:{self.meters.losses.avg:.5f},  current_lr:{current_lr},  previous_loss:{train_loss:.5f}')

                #loss increase, report crash
                if curr_step > 100 and self.meters.losses.avg > train_loss + 0.5:
                    self.logger.info(f'[ERROR] Training Loss Crashed,lr:{current_lr},prec1:{prec1},curr_step:{curr_step}, meters.losses.avg:{self.meters.losses.avg}')
                else:
                    train_loss = self.meters.losses.avg


                self.optimizer.zero_grad()

                def param_clip_before():
                    if self.config.grad_clip.type == 'constant':
                        self.model.module.logit_scale.requires_grad = False
                    elif self.config.grad_clip.type == 'logit_scale_param':
                        before = self.model.module.logit_scale.data.item()
                    elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)
                def param_clip_after():
                    if self.config.grad_clip.type == 'logit_scale_param':
                        after = self.model.module.logit_scale.data.item()
                        tem = self.model.module.logit_scale.data
                        if (after-before) > self.config.grad_clip.value:
                            self.model.module.logit_scale.data = torch.as_tensor(
                                before+self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                        elif (before-after) > self.config.grad_clip.value:
                            self.model.module.logit_scale.data = torch.as_tensor(
                                before-self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                    elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_param_value':  # clip param
                        self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value, max=self.config.grad_clip.max_value)

                def grad_clip_before():  # before update(optimizer.step)
                    if self.config.grad_clip.type == 'norm':
                        clip_grad_norm_(self.model.parameters(),
                                        self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'value':
                        clip_grad_value_(self.model.parameters(),
                                         self.config.grad_clip.value)
                    elif self.config.grad_clip.type == 'logit_scale_grad':
                        clip_param_grad_value_(
                            self.model.module.logit_scale, self.config.grad_clip.value)

                param_clip_before()
                link.barrier()

                loss.backward()
                #self.model.sync_gradients()
                grad_clip_before()
                if self.config.get('check_grad', False):
                    self.check_model_and_grad(10)
                self.optimizer.step()
                link.barrier()
                param_clip_after()

                # clamp
                if self.config.grad_clip.type == 'logit_scale_param_ema':
                    logit_scale.clamp()
                    logit_scale.update()

                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)
                # training logger
                if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                    self.tb_logger.add_scalar(
                        'loss_train', self.meters.losses.avg, curr_step)
                    self.tb_logger.add_scalar(
                        'acc1_train', self.meters.top1.avg, curr_step)
                    self.tb_logger.add_scalar(
                        'acc5_train', self.meters.top5.avg, curr_step)
                    self.tb_logger.add_scalar('lr', current_lr, curr_step)
                    self.tb_logger.add_scalar(
                        'logit_scale_exp', self.model.module.logit_scale.exp(), curr_step)
                    self.tb_logger.add_scalar(
                        'logit_scale', self.model.module.logit_scale, curr_step)
                    self.tb_logger.add_scalar(
                        'delta_logit_scale', self.model.module.logit_scale-last_logit_scale, curr_step)
                    self.tb_logger.add_scalar(
                        'logit_scale_grad', self.model.module.logit_scale.grad, curr_step)
                    self.tb_logger.add_scalar(
                        'clip_number', logit_scale.clip_number, curr_step)

                    remain_secs = (total_step - curr_step) * \
                        self.meters.batch_time.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                    log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                        f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                        f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                        f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                        f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                        f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                        f'LR {current_lr:.4f}\t' \
                        f'logit_scale_exp {float(self.model.module.logit_scale.exp()):.4f}\t' \
                        f'logit_scale {float(self.model.module.logit_scale):.4f}\t' \
                        f'delta_logit_scale {float(self.model.module.logit_scale-last_logit_scale):.4f}\t' \
                        f'logit_scale_grad {float(self.model.module.logit_scale.grad):.4f}\t' \
                        f'clip_number {logit_scale.clip_number:.1f}\t' \
                        f'Remaining Time {remain_time} ({finish_time})'
                    self.logger.critical(log_msg)
                last_logit_scale = self.model.module.logit_scale.clone()

                # testing during training
                if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                    top1_acc = self.evaluate(self.val_data)
                    self.logger.info('Top1 imagenet_acc1_val: {:.3f}'.format(top1_acc))
                    # testing logger
                    if self.dist.rank == 0:
                        self.tb_logger.add_scalar(
                            'imagenet_acc1_val', top1_acc, curr_step)

                #save ckpt---
                if curr_step > 0 and (curr_step % self.config.saver.save_freq == 0 or curr_step == total_step):
                    # save ckpt when at save_freq or the last step !!!
                    if self.dist.rank == 0:
                        if self.config.saver.save_many:
                            ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                        else:
                            ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                        self.state['model'] = self.model.state_dict()
                        self.state['optimizer'] = self.optimizer.state_dict()
                        self.state['last_iter'] = curr_step
                        torch.save(self.state, ckpt_name)

                        if curr_step % (self.config.saver.save_freq*10) == 0:
                            self.logger.info('save model kth')
                            k_times_save_path = f'{self.path.save_path}_k_times'
                            if not os.path.exists(k_times_save_path):
                                os.makedirs(k_times_save_path)
                            ckpt_name = f'{k_times_save_path}/ckpt_{curr_step}.pth.tar'
                            torch.save(self.state, ckpt_name)

                end = time.time()
                if curr_step > total_step:
                    break


    @torch.no_grad()
    def evaluate(self, val_data):
        self.logger.info('start evaluation')

        self.model.eval()

        #---extract prompt embeddings
        label_text, label_text_ensemble_matrix = val_data['loader'].dataset.get_label_texts()
        label_num = label_text_ensemble_matrix.shape[1]
        prompts_num = len(label_text) // label_num
        label_text_preds = []
        for i in range(label_num):
            label_text_pred = self.model.module.encode_text(label_text[i*prompts_num:(i+1)*prompts_num])
            label_text_pred /= (label_text_pred.norm(dim=-1, keepdim=True))
            label_text_pred = label_text_pred.mean(dim=0)
            label_text_pred /= label_text_pred.norm()
            label_text_preds.append(label_text_pred)

        label_text_preds = torch.stack(label_text_preds, dim=0)


        correct_num = 0
        data_num = 0
        #---extract image embeddings and calculatte cosine similarity between image and prompt embeddings
        for batch_idx, batch in enumerate(val_data['loader']):
            #label
            label = batch['label'].to(label_text_preds)

            images = batch['image']
            images = images.cuda()

            #--get img embeddings
            image_preds = self.model.module.encode_image(images)

            #l2 normalize
            image_preds = image_preds / \
                          (image_preds.norm(dim=-1, keepdim=True))

            #cosine sim
            logits = image_preds @ label_text_preds.t()

            #get preds
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)


            # gather output batch information from all gpus
            preds = self.all_gather(preds)
            label = self.all_gather(label)

            assert preds.shape[0] == images.shape[0] * self.dist.world_size

            correct_num += (preds == label).sum().item()
            data_num += preds.shape[0]

        top1_acc = correct_num / data_num * 100

        self.model.train()

        self.logger.info('top1_acc: {:.3f}'.format(top1_acc))
        return top1_acc


def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--batch_size', default=128, type=int)

    args = parser.parse_args()


    init_ddp()

    solver = ClsSolver(args)
    # evaluate or train
    solver.train()


if __name__ == '__main__':
    main()
