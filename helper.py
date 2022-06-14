'''
Author: your name
Date: 2022-04-03 20:05:13
LastEditTime: 2022-04-17 10:14:16
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /code_v4/helper.py
'''

from torch.optim.lr_scheduler import LambdaLR
import torch
import math




def build_optimizer(args, model , train_steps):
    no_decay = ['bias', 'LayerNorm.weight']  # TODO：指哪的bias和哪里的LayerNorm
    # pretrain好的，用小学习率
    pretrain_module = list(model.mylxmert.named_parameters()) 
                        
    # 随机初始化成的，用稍微大的学习率
    other_param_optimizer = list(model.cls.named_parameters())
                            
    optimizer_grouped_parameters = [
        # 除了偏差项和归一化权重，对bert其他参数进行衰减
        ### roberta 是预训练模型，小学习率进行更新
        {'params': [p for n, p in pretrain_module if not any(nd in n for nd in no_decay)],  
         'weight_decay_rate': args.weight_decay, 'lr':args.small_lr},
        {'params': [p for n, p in pretrain_module if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0 , 'lr':args.small_lr},

        # 除了偏差项和归一化权重，对其他网络参数进行衰减
        {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, },  # 应该是单独的学习率
        {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr ,)
    scheduler = MyWarmupCosineSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    return optimizer,scheduler

def build_optimizer_forvilt(args, model , train_steps):
    no_decay = ['bias', 'LayerNorm.weight']  # TODO：指哪的bias和哪里的LayerNorm
    # pretrain好的，用小学习率
    pretrain_module = list(model.vilt.named_parameters()) 
                        
    # 随机初始化成的，用稍微大的学习率
    other_param_optimizer = list(model.cls.named_parameters())
                            
    optimizer_grouped_parameters = [
        # 除了偏差项和归一化权重，对bert其他参数进行衰减
        ### roberta 是预训练模型，小学习率进行更新
        {'params': [p for n, p in pretrain_module if not any(nd in n for nd in no_decay)],  
         'weight_decay_rate': args.weight_decay, 'lr':args.small_lr},
        {'params': [p for n, p in pretrain_module if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0 , 'lr':args.small_lr},

        # 除了偏差项和归一化权重，对其他网络参数进行衰减
        {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, },  # 应该是单独的学习率
        {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr ,)
    scheduler = MyWarmupCosineSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    return optimizer,scheduler

def build_optimizer_fornezha(args, model , train_steps):
    no_decay = ['bias', 'LayerNorm.weight']  # TODO：指哪的bias和哪里的LayerNorm
    # pretrain好的，用小学习率
    pretrain_module = list(model.bert.named_parameters())  + \
        list(model.feats_embedding.named_parameters()) +\
        list(model.layer_norm.named_parameters())     
                        
    # 随机初始化成的，用稍微大的学习率
    other_param_optimizer = list(model.cls.named_parameters())
                            
    optimizer_grouped_parameters = [
        # 除了偏差项和归一化权重，对bert其他参数进行衰减
        ### roberta 是预训练模型，小学习率进行更新
        {'params': [p for n, p in pretrain_module if not any(nd in n for nd in no_decay)],  
         'weight_decay_rate': args.weight_decay, 'lr':args.small_lr},
        {'params': [p for n, p in pretrain_module if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0 , 'lr':args.small_lr},

        # 除了偏差项和归一化权重，对其他网络参数进行衰减
        {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, },  # 应该是单独的学习率
        {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr ,)
    scheduler = MyWarmupCosineSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    return optimizer,scheduler

def build_optimizer_for_allmodels(args , train_steps, pretrain_module, finetune_module):
    no_decay = ['bias', 'LayerNorm.weight']  # TODO：指哪的bias和哪里的LayerNorm
    # pretrain好的，用小学习率  
    pretrain_module = pretrain_module   # pretrain模块，用小学习率 
    finetune_module = finetune_module   # 下游参数,用大学习率                
    optimizer_grouped_parameters = [
        # 除了偏差项和归一化权重，对bert其他参数进行衰减
        ### roberta 是预训练模型，小学习率进行更新
        {'params': [p for n, p in pretrain_module if not any(nd in n for nd in no_decay)],  
         'weight_decay_rate': args.weight_decay, 'lr':args.small_lr},
        {'params': [p for n, p in pretrain_module if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0 , 'lr':args.small_lr},

        # 除了偏差项和归一化权重，对其他网络参数进行衰减
        {'params': [p for n, p in finetune_module if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, },  # 应该是单独的学习率
        {'params': [p for n, p in finetune_module if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr ,)
    scheduler = MyWarmupCosineSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    return optimizer,scheduler



class MyWarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(MyWarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))




