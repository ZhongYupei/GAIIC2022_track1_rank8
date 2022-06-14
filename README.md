# GAIIC2022_track1_rank8

2022人工智能技术创新大赛-赛道1-电商关键属性匹配
[比赛链接](https://www.heywhale.com/home/competition/620b34c41f3cf500170bd6ca/leaderboard)

**成绩**

- 初赛 rank 4
- 复赛 rank 8

另外两名队友

[BaenRH](github.com/BaenRH) & [JoanSF](github.com/JoanSF)


# 题目理解
判断脱敏后的图片数据与文本的属性匹配。

数据来源于电商平台上的商品信息。

特征属性13个类，:

```
图文，领型，袖长，衣长，版型，裙长，穿着方式....

```
针对13个属性做二分类，只需01判断，0表示该图文对不匹配，1表示不匹配。


# 数据前处理
## 数据理解
一个有标签case:

```
{
	img_name = train023876,
	title = 高领灰色休闲男士加厚无扣长袖毛衣,
	key_attr = {领型=高领,袖长=长袖},
	match = {图文=1,领型=1,袖长=1,},
	feature = [-1.1453,-0.00761,...,-0.037] # len 2048
	
}
```

一个无标签case:

```
{
	img_name = train023876,
	title = 高领灰色休闲男士加厚无扣长袖毛衣,
	key_attr = {},
	match = {图文=1},
	feature = [-1.1453,-0.00761,...,-0.037] # len 2048
	
}
```
- **title** :  文本
- **key_attr** : 属性及对应的值，与**title**绑定，从**title**中可反映属性值
- **match** :  属性是否匹配，1表示该case匹配，0表示不匹配
- **feature** : 图片脱敏后的特征数据，长度2048

**key_attr**中的‘图文’ 表示图文是否匹配，如果有某一个属性不匹配，则‘图文’为0

原数据中正样本(图文=1)比例占0.85+，因此需要构造负样本。

无标签的共有10w，有标签有5w。

## 数据前处理

对无标签的10w做了属性值提取（从文本中），转成有标签，最终有标签数据达到13w+。

脱敏特征**feature**官方已处理，均值为0。

删除年份信息，含有年份信息的title占比0.70+，年份信息与模型任务不相关。

## 数据增强

全在**dataset.__getitem__**中增强

###  负样本生成
- 随机title替换
- 替换title中的属性值
- 隐藏属性值替换，如颜色

### 其他增广
- title重组，语义保持不变
- 脱敏特征数据点调换

# 方案

## 模型

bert一把梭

- [LXMERT](https://arxiv.org/pdf/1908.07490.pdf?ref=https://githubhelp.com)
- [VILT](http://proceedings.mlr.press/v139/kim21k/kim21k.pdf)
- [VILBERT](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)

队友用的Nezha+LSTM，Visualbert

## Pretrain

- 任务MLM+ITM
- 13w有标签数据做‘图文’属性ITM和MLM，剩余只做MLM
- 只在‘图文’属性 macth时，才做MLM任务
- 取MLM loss + ITM loss 最低的模型
- LXMERT、VILT、VILBERT 均无使用huggingface预训练权重
- 自己生成的vocab，vocab_size = 1.5k+

## Finetune

### 模型结构
- 2048的feature经过linear转成768。
- 取原模型中的bert backbone，拼接[768,13] linear作为下游进行二分类判断。
- LXMERT  下游 取text 的cross enocder 的pooler_output。
- lxmert 的图片encoder部分去掉了position_embedding和 type_embedding。
- vilbert 下游 text 和 image两个通道 做element wise product 。
- vilt 下游取pooler_output。
- 损失函数：BECLoss。
- 各模型其他输入操作与原文基本一致。
- 三个模型的单模成绩抖动在 $\pm 0.002$ 以内。

### tricks
- 【提升较大】负样本生成。
- 【提升较大】学习率线性warup+余弦衰减。
- 【有提升】nezha+lstm 用了fgm。
- 【有提升】复赛用LXMERT做了多折，对参数进行融合。
- 【提升较大】不同模型的的融合效果大于单个模型的多折效果。
- 【小提升】下游多任务，例如针对属性分类。效果很小，复赛限制了模型规模后就没做考虑。

# 其他尝试

- 【没提升】复赛将数据生成的分布拉成测试集的分布，效果猛降。
- 【没提升】隐藏属性的考虑。考虑了性别，没提升。考虑了颜色，效果不大。

-  【没提升】考虑features已经经过处理，用bert只做文本的特征提取，然后两个特征concat，最后交给一个linear处理。
-  【没提升】离线生成数据集，按照线上测试集的分布，没有提升。可能姿势不对。


# 可考虑的其他trick（没有尝试）

- 对抗训练，只在word embedding加了对抗，feature部分没有考虑
- 一些重复属性的考虑。
- 下游没有做其他尝试。几个模型的下游都是拼接linear，输入是bert的pooler_output。
- 二分类任务，sigmoid出来之后的阈值。
- 双流模型的pretrain任务可以考虑ITC。

# code
pytorch版本
## 文件说明
```
|--- project
	|--- README.md 
	|--- data
		|--- fine_data_sample.json 				# 样例数据集，500个case
	|--- color.txt  							# 颜色词表
	|--- vocab.txt 								# 字典
	|--- jeiba_userdict.txt 					# jieba分词词典
	|--- datasets.py 							# dataset
	|--- helper.py 								# 辅助工具
	|--- lxmert.py 								# model lxmert
	|--- vilt.py 								# model vilt
	|--- vilbert.py 							# model vilbert
	|--- pretrain_lxmert.py 					# lxmert pretrain
	|--- finetune_lxmert.py 					# lxmert finetune
	|--- finetune_lxmert_kfold.py 				# lxmert finetune kfold
	|--- pretrain_vilt.py 						# vilt pretrain
	|--- finetune_vilt.py 						# vilt finetune
	|--- pretrain_vilbert.py 					# vilbert pretrain
	|--- finetune_vilbert.py 					# vilbert finetune
```

## 运行案例
以lxmert为例子，运行样例数据集（`--mode=test`），只有500个case，另外两个模型运行方式类似。

```
## pretrain 
python3 pretrain_lxmert.py \
	--mode test \
	--gpu 0 \
	
## finetune
python3 finetune_lxmert.py \
	--mode test \ 
	--gpu 0
	
## finetune_kfold
python3 finetune_lxmert_kfold.py\
	--mode test\
	--gpu 0
```

# 成绩

--|--|--
初赛B榜|4|0.95211401
复赛B榜|8|0.95045418



# 总结

没有创新点😂，全是bert一把梭哈+搜参微调。

复赛限制了模型scale，所以模型压了一下，成绩有所下降。

第一次参加比赛，重在学习参与，积累学习经验😁。





