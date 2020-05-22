import pandas as pd
import os
import numpy as np
import sys

def kg_data_process():
    train_file='train.txt'
    test_file= 'test.txt'

    n_users,n_items=0,0

    train_len=0

    with open(train_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])

                n_items = max(n_items, max(items))
                n_users = max(n_users, uid)
                train_len+= len(items)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')[1:]]
                except Exception:
                    continue
                n_items = max(n_items, max(items))
    n_items += 1
    n_users += 1

    tp_kg = pd.read_csv('kg_final.txt',sep=' ', header=None)

    tp_kg_inv = tp_kg[[2,1,0]]
    tp_kg_inv.columns = [0,1,2]
    tp_kg = tp_kg.append(tp_kg_inv)
    tp_kg.drop_duplicates(inplace=True)

    head_train = np.array(tp_kg[0], dtype=np.int32)
    relation_train = np.array(tp_kg[1], dtype=np.int32)
    tail_train = np.array(tp_kg[2], dtype=np.int32)

    train_set = {}

    with open(train_file) as f_train:
        for l in f_train.readlines():
            if len(l) == 0: break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            for i in train_items:
                if i in train_set:
                    train_set[i].append(uid)
                else:
                    train_set[i]=[uid]


    relation_set, tail_set = {}, {}

    relation_list = {}
    tail_list = {}
    r_n = 0
    t_n = 0

    f = open('kg_final2.txt', 'w')

    print len(head_train)

    for i in range(len(head_train)):
        if i%100000==0:
            print i
        if head_train[i] in train_set.keys():
            if relation_train[i] not in relation_list.keys():
                relation_list[relation_train[i]]=r_n
                r_n =r_n+1
            if tail_train[i] not in tail_list.keys():
                tail_list[tail_train[i]]=t_n
                t_n=t_n+1

            f.write(str(head_train[i])+' '+str(relation_list[relation_train[i]])+' '+str(tail_list[tail_train[i]])+'\n')


    print r_n, t_n

if __name__ == '__main__':

    kg_data_process()
