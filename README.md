# JNSKR

This is our implementation of the paper:

*Chong Chen, Min Zhang, Weizhi Ma, Yiqun Liu and Shaoping Ma. 2020. [Jointly Non-Sampling Learning for Knowledge Graph Enhanced Recommendation.](https://chenchongthu.github.io/files/SIGIR_JNSKR.pdf) 
In SIGIR'20.*

**Please cite our SIGIR'20 paper if you use our codes. Thanks!**

```
@inproceedings{chen2020jointly,
  title={Jointly Non-Sampling Learning for Knowledge Graph Enhanced Recommendation},
  author={Chen, Chong and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of SIGIR},
  year={2020},
}
```

**You also need to cite the KDD'19 paper if you use the datasets. Thanks!**
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```
Author: Chong Chen (cstchenc@163.com)

## Baselines and Datasets

We follow the previous work, KGAT, and you can get the detailed information about the baselines and datasets in https://github.com/xiangwang1223/knowledge_graph_attention_network. 

## Example to run the codes		

Train and evaluate our model:

```
python main_JNSKR.py
```
Train and evaluate baselines:

```
python main_Baselines.py
```

## Reproducibility

For Amazon dataset:

```
parser.add_argument('--dropout', type=float, default=[0.8,0.7],
                        help='dropout keep_prob')
parser.add_argument('--coefficient', type=float, default=[1.0, 0.01],
                        help='weight of multi-task')
parser.add_argument('--c0', type=float, default=300,
                        help='initial weight of non-observed data')
parser.add_argument('--c1', type=float, default=600,
                        help='initial weight of non-observed knowledge data')
                        
```                        
For Yelp dataset:
```
parser.add_argument('--dropout', type=float, default=[0.9,0.7],
                        help='dropout keep_prob')
parser.add_argument('--coefficient', type=float, default=[1.0, 0.01],
                        help='weight of multi-task')
parser.add_argument('--c0', type=float, default=1000,
                        help='initial weight of non-observed data')
parser.add_argument('--c1', type=float, default=7000,
                        help='initial weight of non-observed knowledge data')

```
## Suggestions for parameters

Several important parameters need to be tuned for different datasets, which are:
```
parser.add_argument('--dropout', type=float, default=[0.8,0.7],
                        help='dropout keep_prob')
parser.add_argument('--coefficient', type=float, default=[1.0, 0.01],
                        help='weight of multi-task')
parser.add_argument('--c0', type=float, default=300,
                        help='initial weight of non-observed data')
parser.add_argument('--c1', type=float, default=600,
                        help='initial weight of non-observed knowledge data')
```
Specifically, c0 and c1 determine the overall weight of non-observed data. The coefficient parameter determines the importance of different tasks in joint learning.

You can also contact us if you can not tune the parameters properly.










