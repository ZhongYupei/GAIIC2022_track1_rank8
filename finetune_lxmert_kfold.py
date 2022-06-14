import time
import numpy as np
import json
import torch
import torch.nn as nn
from copy import deepcopy
from helper import  build_optimizer 
from lxmert import  MyLxmertFinetune
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datasets import MatchDataset_v2 , delete_word
import argparse
from transformers import (
    BertTokenizer,
    set_seed,
)
import os
import gc
import random 


device = 'cuda'
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
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path= opt.tokenizer_path)

    color_set = set()
    with open('./color.txt','r') as file:
        lines = file.readlines()
    for line in lines:
        color_set.add(line[:-1])
   
    with open(os.path.join('./data' ,'sort_label_list.txt'), 'r') as f:
        label_list = [label for label in f.read().strip().split()]
   
    label2id = {key:i for i, key in enumerate(label_list)}

 
    with open(os.path.join('./data','attr_to_attrvals.json'), 'r') as f:
        key_attr_values = json.loads(f.read())
    if opt.mode == 'train':
    
        coarse_train_path = os.path.join(opt.data_root, 'coarse_to_fine_data.json')
        fine_train_path = os.path.join(opt.data_root,'fine_data.json')
    else:
        coarse_train_path = os.path.join(opt.data_root,'fine_data_sample.json')
        fine_train_path = os.path.join(opt.data_root,'fine_data_sample.json')

    since = time.time()
 
    with open(fine_train_path, 'r') as f:
        fine_data = json.loads(f.read())
    fine_texts, fine_img_features, fine_labels, fine_label_masks, fine_key_attrs =\
        fine_data['texts'] , fine_data['img_features'] , fine_data['labels'] , fine_data['label_masks'] , fine_data['key_attrs']
    with open(coarse_train_path, 'r', encoding='utf-8') as f:
        coarse_to_fine_data = json.loads(f.read())
    coarse_to_fine_texts, coarse_to_fine_img_features, coarse_to_fine_labels, coarse_to_fine_label_masks, coarse_to_fine_key_attrs = \
        coarse_to_fine_data['texts'], coarse_to_fine_data['img_features'], coarse_to_fine_data['labels'], coarse_to_fine_data['label_masks'], coarse_to_fine_data['key_attrs']

  
    fine_texts = list(map(delete_word,fine_texts))
    coarse_to_fine_texts = list(map(delete_word,coarse_to_fine_texts))

    data_texts          = np.array(fine_texts + coarse_to_fine_texts)
    data_img_features   = np.array(fine_img_features + coarse_to_fine_img_features)
    data_labels         = np.array(fine_labels + coarse_to_fine_labels)
    data_label_masks    = np.array(fine_label_masks + coarse_to_fine_label_masks)
    data_key_attrs      = np.array(fine_key_attrs + coarse_to_fine_key_attrs)

    folder = KFold(n_splits=opt.kfold,shuffle=False)   # 只对fine_data 进行分折
    splits = folder.split([i for i in range(len(data_texts))])
    
    mylxmert_orgin = MyLxmertFinetune.from_pretrained(opt.pretrain_model_path  , output_dim = 13)


    for fold_idx , (train_idxs , test_idxs) in enumerate(splits):
        torch.cuda.empty_cache()
        train_texts         , test_texts            = data_texts[train_idxs]        , data_texts[test_idxs]
        train_img_features  , test_img_features     = data_img_features[train_idxs] , data_img_features[test_idxs]
        train_labels        , test_labels           = data_labels[train_idxs]       , data_labels[test_idxs]
        train_label_masks   , test_label_masks      = data_label_masks[train_idxs]  , data_label_masks[test_idxs]
        train_key_attrs     , test_key_attrs        = data_key_attrs[train_idxs]    , data_key_attrs[test_idxs]

        train_dataset = MatchDataset_v2(
            tokenizer = tokenizer , 
            texts = train_texts , 
            labels = train_labels , 
            visual_embeds = train_img_features , 
            label_masks = train_label_masks , 
            key_attrs = train_key_attrs , 
            key_attr_values = key_attr_values , 
            label2id = label2id,
            color_set = color_set,
            max_len = max([len(text) for text in train_texts]),
        )
        test_dataset = MatchDataset_v2(
            tokenizer = tokenizer , 
            texts = test_texts , 
            labels = test_labels , 
            visual_embeds = test_img_features , 
            label_masks = test_label_masks , 
            key_attrs = test_key_attrs , 
            key_attr_values = key_attr_values , 
            label2id = label2id,
            p6  = -1,     
            p7  = -1,
            color_set = color_set,
            max_len = max([len(text) for text in test_texts]),
        )
        print('训练集总量 %d 测试集总量 %d'%(len(train_dataset),len(test_dataset)))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
        mylxmert = deepcopy(mylxmert_orgin)
        mylxmert.to(device)

        criterion =  nn.BCELoss()
        total_update_step = opt.epochs * len(train_dataloader)
        optim , scheduler = build_optimizer(opt , mylxmert ,total_update_step )
        record_epoch_arr = []
        best_score , min_loss = float('-inf') , float('inf')
        for epoch in range(opt.epochs):
            since = time.time() 
            mylxmert.train()
            for _ , batch in enumerate(train_dataloader):
                input_ids               = batch['input_ids'].to(device)                 # shape [batch_size,seq_len]
                attention_mask          = batch['attention_mask'].to(device)
                token_type_ids          = batch['token_type_ids'].to(device)            # shape [batch_size,seq_len]
                visual_embeds           = batch['visual_embeds'].to(device)             # shape [batch_size, feat_num, feat_dim]
                visual_attention_mask   = batch['visual_attention_mask'].to(device)     # shape [batch_size, feat_num]
                labels                  = batch['labels'].to(device)                    # shape [batch_size, 13]
                output = mylxmert(
                    input_ids = input_ids,
                    visual_feats = visual_embeds,
                    attention_mask = attention_mask,
                    visual_attention_mask = visual_attention_mask,
                    token_type_ids = token_type_ids,
                )   
                optim.zero_grad()
                imgtxt_loss = criterion(output[:,0],labels[:,0])
                attr_loss = criterion(output[:,1:],labels[:,1:])
                loss = imgtxt_loss + attr_loss
                loss.backward()
                optim.step()
                scheduler.step()
    
            N_img_text , M_img_text = 0, 0
            N_attr , M_attr = 0 , 0
            eval_losses , eval_img_text_losses , eval_attr_losses = [] , [] , []
            with torch.no_grad():
                mylxmert.eval()
                for batch in test_dataloader:
                    input_ids               = batch['input_ids'].to(device)                 # shape [batch_size,seq_len]
                    attention_mask          = batch['attention_mask'].to(device)
                    token_type_ids          = batch['token_type_ids'].to(device)            # shape [batch_size,seq_len]
                    visual_embeds           = batch['visual_embeds'].to(device)             # shape [batch_size, feat_num, feat_dim]
                    visual_attention_mask   = batch['visual_attention_mask'].to(device)     # shape [batch_size, feat_num]
                    labels                  = batch['labels'].to(device)                    # shape [batch_size, 13]    
                    label_masks             = batch['label_masks'].to(device)
                    output = mylxmert(
                        input_ids = input_ids,
                        visual_feats = visual_embeds,
                        attention_mask = attention_mask,
                        visual_attention_mask = visual_attention_mask,
                        token_type_ids = token_type_ids,
                    )   
                    img_txt_logit , attr_logit = output[:,0] , output[:,1:]
                    true_img_text , true_attr = labels[:,0] , labels[:,1:]
       
                    img_text_loss = criterion(img_txt_logit ,true_img_text )
                    attr_loss = criterion(attr_logit ,true_attr )
                    test_loss = img_text_loss + attr_loss
                    eval_losses.append(test_loss)
                    eval_img_text_losses.append(img_text_loss)
                    eval_attr_losses.append(attr_loss)
                   
                    pred_img_text = torch.zeros_like(output[:,0])
                    pred_img_text[output[:,0] >= 0.5] =1
                    N_img_text += torch.sum(pred_img_text == true_img_text).cpu().numpy().item()
                    M_img_text += torch.sum(torch.ones_like(true_img_text)).cpu().numpy().item()
                
                    pred_attr =  torch.zeros_like(output[:,1:])
                    pred_attr[output[:,1:] >= 0.5] = 1
                    pred_attr = pred_attr[label_masks[:, 1:] == 1]
                    true_attr = true_attr[label_masks[:, 1:] == 1]
                    N_attr += torch.sum(pred_attr == true_attr).cpu().numpy().item()
                    M_attr += torch.sum(torch.ones_like(true_attr)).cpu().numpy().item()
                    del input_ids, attention_mask , token_type_ids , visual_embeds , visual_attention_mask , labels , label_masks
                    del output , img_txt_logit , attr_logit , true_img_text , true_attr , img_text_loss , attr_loss , test_loss
                eval_loss = (sum(eval_losses) / len(eval_losses)).detach().cpu().numpy().item()
                eval_img_text_loss = (sum(eval_img_text_losses) / len(eval_img_text_losses)).detach().cpu().numpy().item()
                eval_attr_loss = (sum(eval_attr_losses) / len(eval_attr_losses)).detach().cpu().numpy().item()
                img_text_scores = 0.5 * N_img_text / M_img_text
                attr_scores = 0.5 * N_attr / M_attr
                total_scores = img_text_scores + attr_scores
            is_save_model = 0
            if total_scores > best_score :  
                best_score , is_save_model = total_scores , 1
                save_model(mylxmert,tokenizer , opt ,fold_idx = fold_idx)
            using_time = (time.time() - since) / 60
            print('E%d(t %.2f min) score %.4f (it %.4f attr %.4f) t_l %.6f (it %.6f attr %.6f)  SaveMode %d.'\
                %(epoch , using_time , total_scores , img_text_scores , attr_scores ,eval_loss , eval_img_text_loss ,eval_attr_loss   , is_save_model))
            record_epoch_arr.append([ total_scores , img_text_scores , attr_scores ,eval_loss , eval_img_text_loss ,eval_attr_loss])
        gc.collect()
        torch.cuda.empty_cache()

def save_model(model,tokenizer,opt,fold_idx):
    dir_path = os.path.join(opt.output_root,'kfold-%d/'%(fold_idx))  
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(dir_path, 'pytorch_model.bin'))
    model_to_save.config.to_json_file( os.path.join(dir_path, 'config.json'))
    tokenizer.save_vocabulary(dir_path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type = int, default=2022, help='random seed')
    parser.add_argument('--mode',type=str,default='train',help='train:正式训练; test:代码测试阶段')
    parser.add_argument('--tokenizer_path',type=str,default = './lxmert_model/pretrain/' ,help='tokenizer path') 
    parser.add_argument('--pretrain_model_path',type=str,default = './lxmert_model/pretrain/' ,help='pretrain model path') 
    parser.add_argument('--data_root',type = str , default='./data/',help='数据根路径')
    parser.add_argument('--output_root',type = str , default='./lxmert_model/kfold/',help = '输出根路径' )
    parser.add_argument('--num_workers',type =int,default=16)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--lr',type=float,default=12e-5)
    parser.add_argument('--small_lr',type=float,default=4e-5)   
    parser.add_argument('--batch_size',type = int,default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--gpu',type = int, default=1,help='GPU')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup_ratio')
    parser.add_argument('--kfold',type = int , default=8 , help= '分多少折' )
    opt = parser.parse_args()
    return opt 

def model_param_avg(opt):
    state_dict_list = []
    import collections
    for k in range(opt.kfold):
        state_dict_list.append(torch.load(os.path.join(opt.output_root,'kfold-%d'%(k),'pytorch_model.bin'),map_location = torch.device('cpu')))
    merge_state_dict = collections.OrderedDict()
    for key in state_dict_list[0].keys():
        merge_state_dict[key] = sum([state_dict_list[k][key] for k in range(opt.kfold)]) / opt.kfold
    dir_path = os.path.join(opt.output_root)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(merge_state_dict, os.path.join(dir_path,'pytorch_model.bin'))
    print('融合模型存储路径 : ',dir_path)
    global_model = MyLxmertFinetune.from_pretrained(opt.pretrain_model_path,output_dim = 13) 
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path= opt.tokenizer_path)
    global_model.load_state_dict(merge_state_dict)
    global_model.config.to_json_file( os.path.join(dir_path, 'config.json'))
    tokenizer.save_vocabulary(dir_path)
    print('融合完毕')
    

    
    
if __name__ == '__main__':
    opt = parse_opt()
    print(opt)  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    train(opt)
    model_param_avg(opt)

