import os
import numpy as np
import pandas as pd

def build_train_set(train_set,n_users,relation_set,tail_set, n_relations,n_entities):
    user_lenth = []
    for i in train_set:
        user_lenth.append(len(train_set[i]))
    user_lenth.sort()
    max_user_pi = user_lenth[int(len(user_lenth) * 0.9999)]

    for i in train_set:
        if len(train_set[i]) > max_user_pi:
            train_set[i] = train_set[i][0:max_user_pi]
        while len(train_set[i]) < max_user_pi:
            train_set[i].append(n_users)

    relation_lenth = []
    for i in relation_set:
        relation_lenth.append(len(relation_set[i]))
    relation_lenth.sort()

    max_relation_pi = relation_lenth[int(len(relation_lenth) * 0.9999)]

    for i in relation_set:
        if len(relation_set[i]) > max_relation_pi:
            relation_set[i] = relation_set[i][0:max_relation_pi]
            tail_set[i] = tail_set[i][0:max_relation_pi]
        while len(relation_set[i]) < max_relation_pi:
            relation_set[i].append(n_relations)
            tail_set[i].append(n_entities)
    print max_relation_pi

    return train_set,relation_set,tail_set,max_user_pi,max_relation_pi

def caculate_weight(c0,c1,p,train_set, train_len,relation_train,relation_len):

    m=[0] * len(train_set.keys())
    for i in train_set.keys():
        m[i]=len(train_set[i])*1.0/train_len

    c=[0] * len(train_set.keys())
    tem=0
    for i in train_set.keys():
        tem += np.power(m[i],p)
    for i in train_set.keys():
        c[i]=c0*np.power(m[i],p) / tem


    mk = [0] * len(relation_train.keys())
    for i in relation_train.keys():
        mk[i] = len(relation_train[i]) * 1.0 / relation_len

    ck = [0] * len(relation_train.keys())
    tem = 0
    for i in relation_train.keys():
        tem += np.power(mk[i], p)
    for i in relation_train.keys():
        ck[i] = c1 * np.power(mk[i], p) / tem

    print c[0:10]
    print ck[0:10]
    c = np.array(c)
    ck =  np.array(ck)
    return c,ck

def load_data(DATA_ROOT,args):
    train_file=os.path.join(DATA_ROOT, 'train.txt')
    test_file= os.path.join(DATA_ROOT, 'test.txt')

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

    tp_kg = pd.read_csv(os.path.join(DATA_ROOT, 'kg_final2.txt'),sep=' ', header=None)

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

    for i in range(len(head_train)):

        if head_train[i] in relation_set:
            relation_set[head_train[i]].append(relation_train[i])
            tail_set[head_train[i]].append(tail_train[i])
        else:
            relation_set[head_train[i]] = [relation_train[i]]
            tail_set[head_train[i]] = [tail_train[i]]

    n_relations = max(relation_train) + 1
    n_entities = max(tail_train) + 1
    relation_len =len(head_train)

    negative_c, negative_ck = caculate_weight(args.c0,args.c1,args.p,train_set, train_len,relation_set,relation_len)

    train_set, relation_set, tail_set, max_user_pi, max_relation_pi = \
        build_train_set(train_set,n_users,relation_set,tail_set, n_relations,n_entities)

    return n_users, n_items, n_relations,n_entities, train_set,relation_set, tail_set,max_user_pi,max_relation_pi,negative_c, negative_ck

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def get_train_instances(train_set, relation_set, tail_set):
    item_train, user_train, relation_train1, tail_train1 = [], [], [], []


    for i in relation_set.keys():
        if i in train_set.keys():
            item_train.append(i)
            user_train.append(train_set[i])
            relation_train1.append(relation_set[i])
            tail_train1.append(tail_set[i])
    item_train = np.array(item_train)

    user_train = np.array(user_train)
    relation_train1 = np.array(relation_train1)

    tail_train1 = np.array(tail_train1)

    item_train = item_train[:, np.newaxis]

    return item_train, user_train, relation_train1, tail_train1
def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
