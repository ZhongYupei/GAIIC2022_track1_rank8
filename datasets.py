
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
        p1 = 0.5 ,  # 无关键属性。 用新的title 或者 保留用老的title
        p2 = 0.3 ,   # 有关键属性的数据。  保持不变 发生变化
        p3 = 0.5 ,   # 改变title的概率，直接改变整个标题，否则在原来的标题上作修改
        p4 = 0.95 ,  # 改变文本中的属性文字 / 改变文本中的‘非属性’文字
        p5 = 0.5 ,   # 随机重组title
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
            if random.random() < self.p1 :  # 用新的title
                new_idx = random.choice(range(len(self.labels)))
                while new_idx == idx:
                    new_idx = random.choice(range(len(self.labels)))
                new_text_set = set(list(jieba.cut(self.texts[new_idx])))
                old_text_set = set(list(jieba.cut(self.texts[idx])))
                sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
                sentence_image_labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0
                text = deepcopy(self.texts[new_idx])
            else:       # 保持原来的title
                text = deepcopy(self.texts[idx])
                sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                   dtype=torch.long)
        else:
            # 针对有关键属性的样本
            if random.random() < self.p2:   # 保持不变,用老的title
                text = deepcopy(self.texts[idx])
                sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                   dtype=torch.long)
            else:
                if random.random() < self.p3: # 替换整个标题
                    new_idx = random.choice(range(len(self.labels)))
                    while new_idx == idx :
                        new_idx = random.choice(range(len(self.labels))) 
                    new_text_set = set(list(jieba.cut(self.texts[new_idx])))
                    old_text_set = set(list(jieba.cut(self.texts[idx])))
                    sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)
                    sentence_image_labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0 
                    text = deepcopy(self.texts[new_idx])
                else:
                    if random.random() < self.p4:       # 改变title中的‘属性’文字
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
                        sentence_image_labels = torch.zeros(visual_embeds.shape[:-1], dtype=torch.long)   # 替换了同类型不同属性值，所以一定是不匹配的
                    else:    # 改变title中的‘非属性’文字
                        old_text = deepcopy(self.texts[idx])
                        old_text_set = set(list(jieba.cut(old_text,cut_all=False)))
                        hit_set = old_text_set.intersection(self.color_set) # 存在的颜色词
                        if hit_set is not  None:
                            select_set = self.color_set - hit_set       # 从剩余当中的颜色词进行选择
                            for hit in hit_set :    # 一个title中可能存在多个颜色词
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
                            # 表示没有颜色词，什么修改都没有做
                            text = deepcopy(self.texts[idx])
                            sentence_image_labels = torch.full(visual_embeds.shape[:-1], self.labels[idx][0],
                                                    dtype=torch.long)    
        not_shuffle = '浅' in text or '深' in text or '拼' in text or '撞' in text # 如果有这三个字则不进行打乱
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
        max_len =35,        # 文本的最大长度
        p1 = 0.3 ,          # 产生负样本的操作(小于p1保持正样本；大于p1产生负样本)
        p2 = 0.5 ,          # 替换title(小于p2替换整个文字；大于p2部替换文本中的部分文字)
        p3 = 0.95 ,          # 改变‘属性’/‘非属性’文本(小于p3改变‘属性’文本；大于p3改变‘非属性’文本)
        p4 = -1 ,          # 更改非属性字： 更改性别 # comment 目前设置一定不会更改
        p5 = 1.0 ,          # 更改非属性字： 更改颜色 # comment 有0.05的概率会发生更改
        p6 = 0.2 ,          # 文本打乱
        p8 = 0.5,           # 对无关键属性的负样本增强
        p7 = 0.7,           # feats增强
        shuffle_rate = 0.1,  # feats有‘多少’未知进行调换
        color_set = None,    # 颜色集合
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
        # 相同含义的属性值
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
        if random.random()<self.p7: # 进行部分位置调换，【思考】减少对feats过拟合的作用
            select_ids = np.random.choice([i for i in range(visual_embeds.size(1))],size=int(self.shuffle_rate * visual_embeds.size(1)) , replace=False)
            shuffle_ids = select_ids.copy()
            np.random.shuffle(shuffle_ids)
            while (select_ids == shuffle_ids).sum() ==  len(shuffle_ids):   # 避免shuffle之后还是一样的
                np.random.shuffle(shuffle_ids)
            visual_embeds[:,select_ids] = visual_embeds[:,shuffle_ids]      # 发生调换

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
                labels[0] = 1 if len(new_text_set.intersection(old_text_set)) == len(new_text_set) else 0   # 表明新文本的内容都出现在旧文本中，这时候认为图文匹配
                text = deepcopy(self.texts[new_idx])
            else:   # 不发生更改
                text = deepcopy(self.texts[idx])
        else:
            if random.random() < self.p1:                   # 保持正样本
                pass
            else:# comment 0.7
                if random.random() < self.p2:               # 替换整个titile
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
                    # 在原来的文本上进行修改
                    # 改变‘属性’字体
                    if random.random() <= self.p3:
                        # 改变‘属性’文字
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
                        labels[0] = 0   # 图文匹配为0 
                    else:
                        # 改变‘非属性’字体，改变颜色
                        if random.random() < self.p5:
                            # comment 更换颜色
                            old_text = deepcopy(self.texts[idx])
                            old_text_set = set(list(jieba.cut(old_text , cut_all=False)))
                            hit_set = old_text_set.intersection(self.color_set) # 存在的颜色词
                            if hit_set is not None:
                                select_set = self.color_set - hit_set   # 从剩余当中进行选择
                                for hit in hit_set:         # 一个title中可能有多种颜色，采用循环的方式
                                    select_color = random.choice(list(select_set))
                                    hit_word_set ,select_word_set = set([txt for txt in hit]), set([txt for txt in select_color])
                                    is_overlap = len(hit_word_set.intersection(select_word_set) - {'色'} ) >= 1
                                    while is_overlap:
                                        # comment 保证选择的新词和旧词之间没有重叠的字，避免颜色相似
                                        select_color = random.choice(list(select_set))
                                        hit_word_set , select_word_set = set([txt for txt in hit]), set([txt for txt in select_color])
                                        is_overlap = len(hit_word_set.intersection(select_word_set) - {'色'}) >= 1
                                    select_set = select_set - {select_color}  # 删除已经选择的候选颜色
                                    old_text = old_text.replace(hit,select_color)
                                labels = torch.tensor(self.labels[idx], dtype=torch.float)
                                labels[0]=0     # 更换了颜色，label为0，属性依旧匹配
                                text = old_text
                            else:
                                # 没有颜色词，不发生变换
                                text = deepcopy(self.texts[idx])

        not_shuffle = '浅' in text or '深' in text or '拼' in text or '撞' in text  # 如果有这三个字则不进行打乱
        if not not_shuffle and random.random() < self.p6:
            # 变换title的顺序
            text_arr = list(jieba.cut(text,cut_all=False))     # 对删除了年份的文本进行裁剪
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
   



            

        


        
     