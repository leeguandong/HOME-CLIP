## 中文HOME-CLIP
基于家装家居场景对chineseclip进行微调的模型，效果不错。整体项目clone自 https://github.com/OFA-Sys/Chinese-CLIP

## 1.训练
- **单击多卡训练**
python -m torch.distributed.launch   --nproc_per_node=2   --nnodes=1 --node_rank=0     --master_addr=localhost   --master_port=22222 	main.py        

- **单卡训练**
单卡训练的改动和dalle很相似，多卡时会多处module这个名称

## 2.图文特征提取





## 3.博客介绍

https://blog.csdn.net/u012193416/article/details/125891924





