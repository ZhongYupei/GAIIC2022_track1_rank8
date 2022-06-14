
import random
import torch
import numpy as np
import re
import jieba
from copy import deepcopy

jieba.load_userdict("jieba_userdict.txt")

# 获取同一属性的其他不同含义的属性值
def get_sameattr_values(attr_to_attrvals):
    attrval_sameattr_values = {attr:{} for attr in attr_to_attrvals}
    for attr, values in attr_to_attrvals.items():
        for i in range(len(values)):
            value = values[i]
            sameattr_values = []
            for j in range(len(values)):
                if j == i:
                    continue
                sameattr_value = values[j]
                if '=' in sameattr_value:
                    for v in sameattr_value.split('='):
                        sameattr_values.append(v)
                else:
                    sameattr_values.append(sameattr_value)
            if '=' in value:
                for v in value.split('='):
                    attrval_sameattr_values[attr][v] = sameattr_values
            else:
                attrval_sameattr_values[attr][value] = sameattr_values
    return attrval_sameattr_values

def is_same_mean_attrval(value1, value2, same_mean_attrvals):
    is_same = False
    for values in same_mean_attrvals:
        if value1 in values and value2 in values:
            is_same = True
            break
    return is_same

def delete_word(text):
    text = re.sub(r'\d+年', '', text)
    return text

class PreDataset_v2(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer , 
        texts , 
        visual_embeds , 
        labels ,
        key_attrs , 
        key_attr_values,
        p1 = 0.5 ,  
        p2 = 0.3 ,   
        p3 = 0.5 ,
        p4 = 0.95 ,  
        p5 = 0.5 ,   
        max_len = 35,
        color_set = None,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.visual_embeds = visual_embeds
        self.labels = labels
        self.key_attrs = key_attrs
        self.key_attr_values = key_attr_values

        self.p1 , self.p2 , self.p3 , self.p4 , self.p5 = p1 , p2 , p3 , p4 , p5
        self.max_len = max_len + 2


        self.attrval_sameattr_values = get_sameattr_values(key_attr_values)
        # 相同含义的属性值
        same_mean_attrvals = []
        for values in key_attr_values.values():
            for value in values:
                if '=' in value:
                    same_mean_attrvals.append(value.split('='))
        self.same_mean_attrvals = same_mean_attrvals
        self.color_set = color_set

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        visual_embeds = torch.tensor(self.visual_embeds[idx]).unsqueeze(0)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        key_attrs = deepcopy(self.key_attrs[idx])
        if len(key_attrs) <  1:
            if random.random() < self.p1 :  
                new_idx = random.choice(range(len(self.labels)))
                while new_idx == idx:
                    new_idx = random.choice(range(len(self.labels)))
                new_text_set = set(list(jieba.cut(self.texts[new_idx])))
                old_text_set = set(list(jieba.cut(self.texts[idx])))
                sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
                sentence_image_labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0
                text = deepcopy(self.texts[new_idx])
            else:       
                text = deepcopy(self.texts[idx])
                sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                   dtype=torch.long)
        else:
        
            if random.random() < self.p2: 
                text = deepcopy(self.texts[idx])
                sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                   dtype=torch.long)
            else:
                if random.random() < self.p3: 
                    new_idx = random.choice(range(len(self.labels)))
                    while new_idx == idx :
                        new_idx = random.choice(range(len(self.labels))) 
                    new_text_set = set(list(jieba.cut(self.texts[new_idx])))
                    old_text_set = set(list(jieba.cut(self.texts[idx])))
                    sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
                    sentence_image_labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0 
                    text = deepcopy(self.texts[new_idx])
                else:
                    if random.random() < self.p4:      
                        random_num = random.choice(list(range(len(key_attrs))))
                        randim_idx_list = list(range(len(key_attrs)))
                        random.shuffle(randim_idx_list)
                        random_idx_list = randim_idx_list[:random_num + 1]
                        keys = list(key_attrs.keys())
                        text = deepcopy(self.texts[idx])
                        for random_idx in random_idx_list:
                            random_key = keys[random_idx]
                            value = key_attrs[random_key]
                            random_value = random.choice(list(self.attrval_sameattr_values[random_key][value]))
                            text =  text.replace(value,random_value)
                        sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)   
                    else:    
                        old_text = deepcopy(self.texts[idx])
                        old_text_set = set(list(jieba.cut(old_text,cut_all=False)))
                        hit_set = old_text_set.intersection(self.color_set) 
                        if hit_set is not  None:
                            select_set = self.color_set - hit_set      
                            for hit in hit_set :   
                                select_color = random.choice(list(select_set))
                                hit_word_set , select_word_set = set([txt for txt in hit]) , set([txt for txt in select_color])
                                is_overlap = len(hit_word_set.intersection(select_word_set) - {'色'}) >= 1
                                while is_overlap:
                                    # comment 保证选择的新词和旧词之间没有重叠的字，避免颜色相似
                                    select_color = random.choice(list(select_set))
                                    hit_word_set , select_word_set = set([txt for txt in hit]) , set([txt for txt in select_color])
                                    is_overlap = len(hit_word_set.intersection(select_word_set) - {'色'}) >= 1
                                select_set = select_set - {select_color}
                                old_text = old_text.replace(hit,select_color)
                            text = deepcopy(old_text)
                            sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
                        else:
                            
                            text = deepcopy(self.texts[idx])
                            sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                    dtype=torch.long)    
        not_shuffle = '浅' in text or '深' in text or '拼' in text or '撞' in text
        if not not_shuffle and random.random() < self.p5:
            text_arr = list(jieba.cut(text,cut_all=False))    
            random.shuffle(text_arr)
            text = ''.join(text_arr)   
        inputs = self.tokenizer(text, padding="max_length", max_length=self.max_len, truncation=True)
        item = {key: torch.tensor(val) for key, val in inputs.items()}
        item.update({
            "visual_embeds": visual_embeds,
            "visual_attention_mask": visual_attention_mask,
            'sentence_image_labels':sentence_image_labels,
        })
        return item

class MatchDataset_v2(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer , 
        texts , 
        labels , 
        visual_embeds,
        label_masks,
        key_attrs,
        key_attr_values,
        label2id,
        max_len =35,       
        p1 = 0.3 ,         
        p2 = 0.5 ,          
        p3 = 0.95 ,         
        p4 = -1 ,          
        p5 = 1.0 ,          
        p6 = 0.2 ,        
        p8 = 0.5,         
        p7 = 0.7,          
        shuffle_rate = 0.1, 
        color_set = None,   
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.visual_embeds = visual_embeds
        self.label_masks = label_masks
        self.key_attrs = key_attrs
        
        self.max_len = max_len + 2
        self.attrval_sameattr_values = get_sameattr_values(key_attr_values)
        self.label2id = label2id
        
        same_mean_attrvals = []
        for values in key_attr_values.values():
            for value in values:
                if '=' in value:
                    same_mean_attrvals.append(value.split('='))
        self.same_mean_attrvals = same_mean_attrvals
        self.color_set = color_set

        # comment 文本增强
        self.p1 , self.p2 , self.p3 , self.p4 , self.p5 , self.p6 , self.p8  = p1 , p2 , p3 , p4 , p5 , p6 , p8
        # comment feats增强
        self.p7 , self.shuffle_rate = p7 , shuffle_rate
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        visual_embeds = torch.tensor(self.visual_embeds[idx], dtype=torch.float32).unsqueeze(0) # shape[1,2048]
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        # comment feats增强
        if random.random()<self.p7: 
            select_ids = np.random.choice([i for i in range(visual_embeds.size(1))],size=int(self.shuffle_rate * visual_embeds.size(1)) , replace=False)
            shuffle_ids = select_ids.copy()
            np.random.shuffle(shuffle_ids)
            while (select_ids == shuffle_ids).sum() ==  len(shuffle_ids):  
                np.random.shuffle(shuffle_ids)
            visual_embeds[:,select_ids] = visual_embeds[:,shuffle_ids]   

        # comment 文本增强 
        # TODO 考虑是否要进行 '删字'
        text = deepcopy(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        label_masks = torch.tensor(self.label_masks[idx])
        key_attrs = deepcopy(self.key_attrs[idx])
        if len(key_attrs) < 1:  
            if random.random() < self.p8:
                new_idx = random.choice(range(len(self.labels)))
                while new_idx == idx:
                    new_idx = random.choice(range(len(self.labels)))
                new_text_set = set(list(jieba.cut(self.texts[new_idx])))
                old_text_set = set(list(jieba.cut(self.texts[idx])))
                labels = torch.zeros(13) 
                labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0   
                text = deepcopy(self.texts[new_idx])
            else:  
                text = deepcopy(self.texts[idx])
        else:
            if random.random() < self.p1:                  
                pass
            else:# comment 0.7
                if random.random() < self.p2:             
                    new_idx = random.choice(range(len(self.labels)))
                    while new_idx == idx :
                        new_idx = random.choice(range(len(self.labels)))
                    new_text , new_keyattrs = deepcopy(self.texts[new_idx]) , deepcopy(self.key_attrs[new_idx])
                    old_text , old_keyattrs = deepcopy(self.texts[idx]) , deepcopy(self.key_attrs[idx])

                    labels = torch.zeros(13)
                    for attr , value in old_keyattrs.items():
                        if attr in new_keyattrs.keys():
                            if value == new_keyattrs[attr]:
                                labels[self.label2id[attr]] = 1
                            if is_same_mean_attrval(value,new_keyattrs[attr],self.same_mean_attrvals):
                                labels[self.label2id[attr]] = 1
                    label_masks = torch.tensor(self.label_masks[new_idx])
                    # comment 判断old_text的文本是否全部出现在new_text中
                    # comment 如果flag为1表示新文本的所有内容出现在旧文本中，这时候label[0] =1
                    # comment 如果flag为0表示新文本中有部分内容没有出现在旧文本中，这时候label[0]=-
                    new_text_set = set(list(jieba.cut(new_text,cut_all=False)))
                    old_text_set = set(list(jieba.cut(old_text,cut_all=False)))
                    labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0
                    text = new_text
                else:
                    # comment 0.7
                   
                    if random.random() <= self.p3:
                        
                        text = deepcopy(self.texts[idx] )
                        random_num = random.choice(list(range(len(key_attrs))))
                        random_idx_list = list(range(len(key_attrs)))
                        random.shuffle(random_idx_list)
                        random_idx_list = random_idx_list[:random_num+1 ]
                        keys = list(key_attrs.keys())
                        text = deepcopy(self.texts[idx])
                        for random_idx in random_idx_list:
                            random_key = keys[random_idx]
                            value = key_attrs[random_key]
                            random_value = random.choice(list(self.attrval_sameattr_values[random_key][value]))
                            text = text.replace(value,random_value)
                            labels[self.label2id[random_key]]= 0
                        labels[0] = 0   
                    else:
                        
                        if random.random() < self.p5:
                            # comment 更换颜色
                            old_text = deepcopy(self.texts[idx])
                            old_text_set = set(list(jieba.cut(old_text , cut_all=False)))
                            hit_set = old_text_set.intersection(self.color_set) 
                            if hit_set is not None:
                                select_set = self.color_set - hit_set  
                                for hit in hit_set:        
                                    select_color = random.choice(list(select_set))
                                    hit_word_set ,select_word_set = set([txt for txt in hit]), set([txt for txt in select_color])
                                    is_overlap = len(hit_word_set.intersection(select_word_set) - {'色'} ) >= 1
                                    while is_overlap:
                                        # comment 保证选择的新词和旧词之间没有重叠的字，避免颜色相似
                                        select_color = random.choice(list(select_set))
                                        hit_word_set , select_word_set = set([txt for txt in hit]), set([txt for txt in select_color])
                                        is_overlap = len(hit_word_set.intersection(select_word_set) - {'色'}) >= 1
                                    select_set = select_set - {select_color}  
                                    old_text = old_text.replace(hit,select_color)
                                labels = torch.tensor(self.labels[idx], dtype=torch.float)
                                labels[0]=0     
                                text = old_text
                            else:
                              
                                text = deepcopy(self.texts[idx])

        not_shuffle = '浅' in text or '深' in text or '拼' in text or '撞' in text 
        if not not_shuffle and random.random() < self.p6:
            text_arr = list(jieba.cut(text,cut_all=False))     
            random.shuffle(text_arr)
            text = ''.join(text_arr)
        inputs = self.tokenizer(text, padding="max_length", max_length=self.max_len, truncation=True)
        item = {key : torch.tensor(val) for key , val in inputs.items()}
        item.update({
            "labels": labels,
            "visual_embeds": visual_embeds,
            "visual_attention_mask": visual_attention_mask,
            "label_masks": label_masks,
        })

        return item
   



            

        


        
     