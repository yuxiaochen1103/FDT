import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import json
import prototype.linklink as link


from prototype.solver.base_solver import BaseSolver
from prototype.utils.torch_ddp_dist import get_rank, get_world_size, get_local_rank, set_random_seed, init_ddp, convert_to_ddp_model
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, \
    param_group_all, AverageMeter, accuracy, load_state_optimizer, \
    parse_config
from prototype.model import model_entry
from prototype.optimizer import optim_entry
from prototype.lr_scheduler import scheduler_entry
from prototype.data.datasets.clip_dataset_wsd import get_wds_dataset
from prototype.loss_functions import LabelSmoothCELoss, ClipInfoCELoss
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_


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
        self.prototype_info = EasyDict() #a dict
        self.config = parse_config(args.config)
        #update config from command lines

        #learning rate
        #self.config.lr_scheduler.kwargs.base_lr = self.args.base_lr
        #self.config.lr_scheduler.kwargs.warmup_lr = self.args.warmup_lr

        #model parameters
        # self.config.model.kwargs.fdt.sd_temperature = self.args.sd_T
        # self.config.model.kwargs.fdt.att_func_type = self.args.att_func_type
        # self.config.model.kwargs.fdt.pool_type = self.args.pool_type
        # self.config.model.kwargs.fdt.sd_num = self.args.sd_num
        # self.config.model.kwargs.fdt.sd_dim = self.args.sd_dim

        #temperature decay schedule
        # self.config.t_decay.org_t = self.args.sd_T
        # self.config.t_decay.sd_T_decay_iter = self.args.sd_T_decay_iter #iteration
        # self.config.t_decay.sd_T_decay_w = self.args.sd_T_decay_w #decay weight
        # self.config.t_decay.sd_T_min = self.args.sd_T_min #min

        #output path
        self.config.output_path = args.output_path


        #---training dataset
        self.config.data.train.batch_size = args.batch_size

        #set env
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        #---- (1)set ddp env (2) set random seed (3)  set output directories


        #set random seed
        set_random_seed()
        self.dist = EasyDict()

        #----get local rank, global rank and world size
        self.dist.local_rank, self.dist.rank, self.dist.world_size = get_local_rank(), get_rank(), get_world_size()
        self.prototype_info.world_size = self.dist.world_size

        #-----set output directories
        self.path = EasyDict()
        self.path.output_path = self.config.output_path

        ft_cfg = self.config.model.kwargs.fdt
        decay_cfg = self.config.t_decay
        self.path.output_path = self.path.output_path + \
                                         '/sd-num-{}_sd-dim-{}_warmup-lr-{}_pool-{}_sd-T-{}_T-decay-w-{}_T-min-{}_T-iter-{}'.format( \
                                             ft_cfg.sd_num, ft_cfg.sd_dim,
                                             self.config.lr_scheduler.kwargs.warmup_lr,
                                             ft_cfg.pool_type,
                                             ft_cfg.sd_temperature, decay_cfg.sd_T_decay_w, decay_cfg.sd_T_min, decay_cfg.sd_T_decay_iter
                                         )

        self.path.save_path = os.path.join(self.path.output_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.output_path, 'events')
        self.path.result_path = os.path.join(self.path.output_path, 'results') #save location as the config_file

        #make local dir
        makedir(self.path.output_path)
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)



        # create logger
        self.path.log_path = os.path.join(self.path.output_path, 'log.txt')
        create_logger(self.path.log_path) #local
        self.logger = get_logger(__name__)

        #--------------
        # add host names
        self.logger.critical(f'config: {pprint.pformat(self.config)}')
        #----------------------



        # create tb_logger for the main process
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)

            #save config file
            config_pth = self.path.output_path + '/config.json'
            with open(config_pth, 'w') as file:
                json.dump(self.config, file)

            self.logger.critical('saved configs to {}'.format(self.config.output_path + '/config.json'))



        self.state = {}
        self.state['last_iter'] = 0

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

        # split parameters to different group, and for different groups, using different paramters
        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        #----optimizer
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

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




    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        self.criterion = ClipInfoCELoss()



    def train(self):
        self.pre_train() #set up for setting

        dataloader = self.train_data.dataloader
        each_epoch_step = dataloader.num_batches
        epoch = self.config.data.train.epoch
        total_step = epoch * each_epoch_step


        self.logger.info('each_epoch_step: {} total_step: {}'.format(each_epoch_step, total_step))

        start_step = self.state['last_iter']
        curr_step = start_step
        end = time.time()
        last_logit_scale = 0
        logit_scale = EMA_logit_scale(self.model.module.logit_scale,
                          self.config.grad_clip.value)
        train_loss = 1e9

        for epoch_id in range(epoch):
            self.train_data.set_epoch(epoch_id) #set epoch
            for i, (image, text) in enumerate(dataloader):
                curr_step += 1
                self.lr_scheduler.step(curr_step) #learning rate is calculated based on step

                if curr_step % self.config.t_decay.sd_T_decay_iter == 0:

                    sd_T = self.config.t_decay.org_t #get orginal temperature
                    sd_T_decay_w = self.config.t_decay.sd_T_decay_w #get decay weight
                    sd_T_decay_iter = self.config.t_decay.sd_T_decay_iter #get decay iter
                    sd_T_min = self.config.t_decay.sd_T_min #get min temperature

                    temperature = sd_T * (sd_T_decay_w ** (curr_step / sd_T_decay_iter))
                    temperature = max(temperature, sd_T_min)

                    self.model.module.img_query_model.temperature = temperature
                    self.model.module.txt_query_model.temperature = temperature


                current_lr = self.lr_scheduler.get_lr()[0]
                # measure data loading time
                self.meters.data_time.update(time.time() - end)

                image = image.cuda()

                logit_sd, _ = self.model(image, text) #did it fuse multi-gpu>

                loss, target = self.criterion(logit_sd[0], logit_sd[1])
                loss = loss / self.dist.world_size
                prec1, prec5 = accuracy(logit_sd[0], target, topk=(1, self.topk))



                #-----update meter
                reduced_loss = loss.clone()
                reduced_prec1 = prec1.clone() / self.dist.world_size
                reduced_prec5 = prec5.clone() / self.dist.world_size


                self.meters.losses.reduce_update(reduced_loss)
                self.meters.top1.reduce_update(reduced_prec1)
                self.meters.top5.reduce_update(reduced_prec5)


                if curr_step % 50 == 0:
                    self.logger.info(f'Epoch[{epoch_id+1}] Iter[{curr_step}]: '
                                     f'losses.avg:{self.meters.losses.avg:.5f},  '
                                     f'current_lr:{current_lr},  previous_loss:{train_loss:.5f} '
                                     f'temperature:{self.model.module.img_query_model.temperature:.5f}'
                                     )

                #loss increase, report crash
                if curr_step > 100 and self.meters.losses.avg > train_loss + 0.5:
                    resume = True
                    self.logger.info(f'[ERROR] Training Loss Crashed,lr:{current_lr},prec1:{prec1},curr_step:{curr_step}, meters.losses.avg:{self.meters.losses.avg}')
                else:
                    train_loss = self.meters.losses.avg

                #self.logger.info('self.dist.rank', self.dist.rank, 'forward done')
                # compute and update gradient


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
                        #for sd
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
                        #for sd


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
                def grad_clip_after():
                    pass

                #self.logger.info('self.dist.rank', self.dist.rank, 'clip grad begin')
                param_clip_before()
                link.barrier()

                loss.backward()
                #self.model.sync_gradients()
                grad_clip_before()
                if self.config.get('check_grad', False):
                    self.check_model_and_grad(10)
                self.optimizer.step()
                grad_clip_after()
                link.barrier()
                param_clip_after()

                # clamp
                if self.config.grad_clip.type == 'logit_scale_param_ema':
                    logit_scale.clamp()
                    logit_scale.update()
                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)
                # training logger

                #self.logger.info('self.dist.rank', self.dist.rank, 'save log')
                if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:

                    #---------------
                    self.tb_logger.add_scalar(
                        'loss_all', self.meters.losses.avg, curr_step)
                    self.tb_logger.add_scalar(
                        'acc1_train', self.meters.top1.avg, curr_step)
                    self.tb_logger.add_scalar(
                        'acc5_train', self.meters.top5.avg, curr_step)
                    self.tb_logger.add_scalar('lr', current_lr, curr_step)
                    self.tb_logger.add_scalar(
                        'logit_scale_exp', self.model.module.logit_scale.exp(), curr_step)
                    self.tb_logger.add_scalar(
                        'logit_scale', self.model.module.logit_scale, curr_step)
                    # ---------------
                    self.tb_logger.add_scalar(
                        'delta_logit_scale', self.model.module.logit_scale-last_logit_scale, curr_step)
                    if self.model.module.logit_scale.grad is not None:
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
                        f'Loss_all {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                        f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                        f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                        f'LR {current_lr:.4f}\t' \
                        f'logit_scale_exp {float(self.model.module.logit_scale.exp()):.4f}\t' \
                        f'logit_scale {float(self.model.module.logit_scale):.4f}\t' \
                        f'delta_logit_scale {float(self.model.module.logit_scale-last_logit_scale):.4f}\t' \
                        f'clip_number {logit_scale.clip_number:.1f}\t' \
                        f'Remaining Time {remain_time} ({finish_time})'
                    self.logger.critical(log_msg)
                last_logit_scale = self.model.module.logit_scale.clone()

                #self.logger.info('self.dist.rank', self.dist.rank, 'save done')


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
                # if curr_step > total_step:
                #     break


def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    
    #output pt
    parser.add_argument('--output_path', required=True, type=str)

    parser.add_argument('--batch_size', default=128, type=int)


    # parser.add_argument('--sd_num', default=16384, type=int) #fdt nums
    # parser.add_argument('--sd_dim', default=512, type=int) #fdt dims
    # parser.add_argument('--att_func_type', default='sparsemax', type=str) #normalization function of attention weights ['sparsemax', 'softmax']
    # parser.add_argument('--pool_type', default='max', type=str) #pooing type attention weights ['max', 'mean']
    #
    #
    # parser.add_argument('--sd_T', default= 1000, type=float) #the tempture parameters of the attention weights
    # parser.add_argument('--sd_T_decay_w', default= 0.3, type=float) #decay ratio of parameters
    # parser.add_argument('--sd_T_decay_iter', default= 2700, type=float) #decay at every sd_T_decay_iter iterations
    # parser.add_argument('--sd_T_min', default= 0.01, type=float) #min value of sd_T
    #
    # parser.add_argument('--base_lr', default= 0.0001, type=float) #inital lr
    # parser.add_argument('--warmup_lr', default= 0.001, type=float) #warmup lr

    args = parser.parse_args()

    # set up pytorch ddp
    init_ddp()

    solver = ClsSolver(args)
    # evaluate or train
    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
