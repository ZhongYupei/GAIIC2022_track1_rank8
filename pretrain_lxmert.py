

import time
import numpy as np
import json
from transformers import (
    BertTokenizer, 
    set_seed,
    DataCollatorForLanguageModeling,
)
from sklearn.model_selection import train_test_split
from transformers import LxmertConfig
from transformers import LxmertTokenizer
from torch.utils.data import DataLoader

import torch
from lxmert import MyLxmertForPreTraining
import argparse
import os
from datasets import *

device = "cuda"
import  random
def seed_everything(seed):
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def train(opt):
    seed_everything(opt.seed)

    # 加载颜色词
    
    color_set = set()
    with open('./color.txt','r') as file:
        lines = file.readlines()
    for line in lines:
        color_set.add(line[:-1])
    color_set = color_set - {''}

    with open(os.path.join('./data' ,'sort_label_list.txt'), 'r') as f:
        label_list = [label for label in f.read().strip().split()]
    # 获取属性的所有值
    with open(os.path.join('./data','attr_to_attrvals.json'), 'r') as f:
        key_attr_values = json.loads(f.read())
    # 文件路径
    t1 = time.time()
    if opt.mode == 'train':
        # 训练样本
        coarse_train_path = os.path.join(opt.data_root,'coarse_data.json')
        fine_train_path = os.path.join(opt.data_root,'fine_data.json')
    else:
        # 测试样本
        coarse_train_path = os.path.join(opt.data_root,'fine_data_sample.json')
        fine_train_path = os.path.join(opt.data_root,'fine_data_sample.json')
    print('*'*50,' Load Data ','*'*50)
    since = time.time()
    with open(coarse_train_path, 'r', encoding='utf-8') as f:
        coarse_data = json.loads(f.read())
    with open(fine_train_path, 'r', encoding='utf-8') as f:
        fine_data = json.loads(f.read())
    # 获取数据
    print('读取数据花费的时间:', time.time() - t1)
    fine_texts, fine_img_features, fine_labels = fine_data['texts'], fine_data['img_features'], fine_data['labels']
    coarse_texts, coarse_img_features, coarse_labels = coarse_data['texts'], coarse_data['img_features'], coarse_data['labels']
    fine_key_attrs = fine_data['key_attrs']
    coarse_key_attrs = coarse_data['key_attrs']
    # 对texts进行裁剪
    fine_texts = [delete_word(text) for text in fine_texts]
    coarse_texts = [delete_word(text) for text in coarse_texts]

    data_texts = np.array(fine_texts  + coarse_texts )
    data_img_features = np.array(fine_img_features  + coarse_img_features )
    data_labels = np.array(fine_labels  + coarse_labels )
    data_key_attrs = np.array(fine_key_attrs  + coarse_key_attrs )
    assert 0 <= opt.test_rate < 1
    # 如果test_size为0，即全部数据一起训练
    if opt.test_rate == 0:
        _, test_idxs = train_test_split(range(len(data_texts)), test_size=0.3)
        train_idxs = [i for i in range(len(data_texts))]
    else:
        # 划分训练集和测试集
        train_idxs, test_idxs = train_test_split(range(len(data_texts)), test_size=opt.test_rate)

    train_texts = data_texts[train_idxs]
    train_img_features = data_img_features[train_idxs]
    train_labels = data_labels[train_idxs]
    train_key_attrs = data_key_attrs[train_idxs]
    test_texts = data_texts[test_idxs]
    test_img_features = data_img_features[test_idxs]
    test_labels = data_labels[test_idxs]
    test_key_attrs = data_key_attrs[test_idxs]

    print('构造数据集合完成。训练集合 %d 测试集合 %d'%(len(train_texts),len(test_texts)))
    tokenizer = LxmertTokenizer.from_pretrained(opt.tokenizer_path)     
    train_dataset = PreDataset_v2(
        tokenizer = tokenizer ,
        texts = train_texts , 
        visual_embeds = train_img_features ,
        labels = train_labels ,
        key_attrs = train_key_attrs ,
        key_attr_values = key_attr_values ,
        color_set = color_set,
    )

    test_dataset = PreDataset_v2(
        tokenizer = tokenizer ,
        texts = test_texts,
        visual_embeds = test_img_features,
        labels = test_labels,
        key_attrs = test_key_attrs ,
        key_attr_values = key_attr_values,
        color_set = color_set,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # DataLoaders creation:
    train_dataloader = DataLoader(train_dataset , shuffle=True , collate_fn = data_collator , batch_size = opt.batch_size, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset , shuffle=False , collate_fn = data_collator , batch_size = opt.batch_size , num_workers=opt.num_workers)
    

    config = LxmertConfig(
        vocab_size = tokenizer.vocab_size,
        task_matched=True,              # 做 MLM 任务
        task_mask_lm=True,              # 做 匹配 任务
        task_obj_predict=False,
        task_qa=False,
        visual_obj_loss=False,
        visual_attr_loss=False,
        visual_feat_loss=False,
        l_layers = opt.l_layer,  
        r_layers = opt.r_layer,  
        x_layers = opt.x_layer,  
    )
    pretrain_mylxmert = MyLxmertForPreTraining(config)
    optim = torch.optim.AdamW(pretrain_mylxmert.parameters(), lr=opt.lr,betas=(0.95,0.999),weight_decay=1e-4)   # TODO 用余弦衰减率
    pretrain_mylxmert = torch.nn.parallel.DataParallel(pretrain_mylxmert.to(device))
    # pretrain_mylxmert.to(device)

    
    

    record_epoch_arr = []
    min_loss = float('inf')
    for epoch in range(opt.epochs):
        since = time.time()
        pretrain_mylxmert.train()
        for _, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)                           # shape[batch_size, seq_len] 
            attention_mask = batch['attention_mask'].to(device)                 # shape[batch_size, seq_len] 
            token_type_ids = batch['token_type_ids'].to(device)                 # shape[batch_size, seq_len] 
            visual_embeds = batch['visual_embeds'].to(device)                   # shape[batch_size, 1 ,2048]      
            visual_attention_mask = batch['visual_attention_mask'].to(device)   # shape[batch_size, 1]      
            is_pared = batch['sentence_image_labels'].to(device)                # shape[batch_size, 1]      是否图文匹配
            true_mlm_text = batch['labels'].to(device)                          # shape[batch_size, seq_len]   真实文本的标签
            output_dict = pretrain_mylxmert(
                input_ids = input_ids , 
                attention_mask = attention_mask,
                token_type_ids = token_type_ids, 
                visual_feats = visual_embeds ,
                visual_attention_mask  = visual_attention_mask, 
                is_paired  = is_pared, # 图文匹配标签
                mlm_true_label = true_mlm_text,  # 文本标签 用于MLM
            )
            mlm_loss , match_loss = output_dict['mlm_loss'],output_dict['match_loss']
            loss = mlm_loss + match_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            

        with torch.no_grad():
            pretrain_mylxmert.eval()
            count  , test_mlm_losses , test_match_losses  = 0  , 0.0 , 0.0
            total_right_num , total_num = 0 , 0
            for _ , batch in enumerate(test_dataloader):
                input_ids = batch['input_ids'].to(device)                           # shape[batch_size, seq_len] 
                attention_mask = batch['attention_mask'].to(device)                 # shape[batch_size, seq_len] 
                token_type_ids = batch['token_type_ids'].to(device)                 # shape[batch_size, seq_len] 
                visual_embeds = batch['visual_embeds'].to(device)                   # shape[batch_size, 1 ,2048]      
                visual_attention_mask = batch['visual_attention_mask'].to(device)   # shape[batch_size, 1]      
                is_pared = batch['sentence_image_labels'].to(device)                # shape[batch_size, 1]      是否图文匹配
                true_mlm_text = batch['labels'].to(device)                          # shape[batch_size, seq_len]   真实文本的标签

                output_dict = pretrain_mylxmert(
                    input_ids = input_ids , 
                    attention_mask = attention_mask,
                    token_type_ids = token_type_ids, 
                    visual_feats = visual_embeds ,
                    visual_attention_mask  = visual_attention_mask, 
                    is_paired  = is_pared, # 图文匹配标签
                    mlm_true_label = true_mlm_text,  # 文本标签 用于MLM
                )
                right_match , mlm_loss , match_loss =  output_dict['right_match'],output_dict['mlm_loss'],output_dict['match_loss']
                total_right_num += right_match
                total_num += len(is_pared)
                
                test_mlm_losses += mlm_loss.detach().cpu().numpy().item()
                test_match_losses += match_loss.cpu().numpy().item()
                count += 1
        test_mlm_losses /= count
        test_match_losses /= count
        correct_rate = total_right_num /total_num 
        record_epoch_arr.append([ correct_rate, test_mlm_losses , test_match_losses ])
        
        using_time = (time.time() - since)  /60
        # 选择testloss最小的epoch保存模型
        current_loss , is_save_model =  test_match_losses + test_mlm_losses , 0
        if current_loss < min_loss:
            min_loss = current_loss
            is_save_model =1 
            save_model(pretrain_mylxmert , tokenizer , opt)
        print('Epoch %d (t %.2f min) correct_rate %.6f mlm_l %.6f match_l %.6f SaveMode %d.'%(epoch,using_time, correct_rate,test_mlm_losses,test_match_losses,is_save_model))

    # 保存模型保存记录结果
    strings = time.strftime('%Y,%m,%d,%H,%M,%S')
    t = strings.split(',')
    number = [int(i) for i in t]
    dir_path = os.path.join(opt.output_root)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # 保存训练记录
    full_path = os.path.join(dir_path,'static-%02d%02d.npy'%(number[3],number[4]))
    np.save(full_path,record_epoch_arr)

    write_opt(opt)

def write_opt(opt):
    # 写入opt文件
    params = [attr for attr in dir(opt) if not attr.startswith('_')]
    content = ''
    for param in params:
        content += '%s : %s \n'%(param,opt.__getattribute__(param))
    strings = time.strftime('%Y,%m,%d,%H,%M,%S')
    t = strings.split(',')
    number = [int(i) for i in t]
    dir_path = os.path.join(opt.__getattribute__('output_root'))
    
    os.makedirs(dir_path , exist_ok=True)
    with open(os.path.join(dir_path,'opt_parm.txt'),'w+',encoding='utf-8') as file:
        file.write(content)

def save_model(model,tokenizer,opt):
    strings = time.strftime('%Y,%m,%d,%H,%M,%S')
    t = strings.split(',')
    number = [int(i) for i in t]
    dir_path = os.path.join(opt.output_root)  

    os.makedirs(dir_path,exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(dir_path, 'pytorch_model.bin'))
    model_to_save.config.to_json_file( os.path.join(dir_path, 'config.json'))
    tokenizer.save_vocabulary(dir_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='train',help='train:正式训练; test:代码测试阶段')
    parser.add_argument('--seed',type = int, default=2022, help='random seed')
    parser.add_argument('--tokenizer_path',type=str,default='./',help='tokenizer path') # comment 使用自己创建的字典
    parser.add_argument('--data_root',type = str , default='./data/',help='数据根路径')
    parser.add_argument('--output_root',type = str , default='./lxmert_model/pretrain/',help = '输出根路径' )

    parser.add_argument('--num_workers',type =int,default=16)
    parser.add_argument('--test_rate',type = float,default=0.1,help='测试集的比例')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--lr',type=float,default=1e-4) # TODO
    parser.add_argument('--batch_size',type = int,default=512)
    parser.add_argument('--gpu',type = int, default=0,help='GPU')

    parser.add_argument('--r_layer',type = int,default=2,help='r_layer')
    parser.add_argument('--x_layer',type = int,default=3,help='x_layer')
    parser.add_argument('--l_layer',type = int,default=5,help='l_layer')


    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    train(opt)


