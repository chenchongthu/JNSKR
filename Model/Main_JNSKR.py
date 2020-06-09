import tensorflow as tf
from utility.our_test import *
import time
import argparse
from JNSKR import JNSKR
import sys
from utility.our_helper import *

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

def parse_args():
    parser = argparse.ArgumentParser(description="Run JNSKR")
    
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=101,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=[0.8,0.7],
                        help='dropout keep_prob')
    parser.add_argument('--coefficient', type=float, default=[1.0, 0.01],
                        help='weight of multi-task')
    parser.add_argument('--lambda_bilinear', type=float, default=[1e-5, 1],
                        help='weight of regul')
    parser.add_argument('--c0', type=float, default=300,
                        help='initial weight of non-observed data')
    parser.add_argument('--c1', type=float, default=600,
                        help='initial weight of non-observed knowledge data')
    parser.add_argument('--p', type=float, default=0.5,
                        help='significance level of weight')
    parser.add_argument('--sparsity', type=float, default=0,
                        help='sparsity test')
    return parser.parse_args()


def train_step2(u_batch, i_batch, r_batch, t_batch, args):
    """
    A single training step
    """

    feed_dict = {
        model.input_i: i_batch,
        model.input_iu: u_batch,
        model.input_hr: r_batch,
        model.input_ht: t_batch,
        model.dropout_keep_prob: args.dropout[0],
        model.dropout_kg: args.dropout[1],
    }

    _, loss, loss1, loss2,w= sess.run(
        [optimizer1, model.loss, model.loss1, model.loss2,model.entities_w],
        feed_dict)

    #print W[i_batch]
    return loss, loss1, loss2,w


if __name__ == '__main__':
    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()
    DATA_ROOT = '../Data/amazon-book'
    f1 = open(os.path.join(DATA_ROOT, 'JNSKR.txt'), 'w')
    n_users, n_items, n_relations, n_entities,train_set, relation_set, tail_set,max_user_pi,max_relation_pi,negative_c, negative_ck\
        =load_data(DATA_ROOT,args)
    print n_users, n_items, n_relations, n_entities, max_user_pi,max_relation_pi

    batch_size = args.batch_size
    lr = args.lr
    embedding_size = args.embed_size
    epochs = args.epochs

    item_train, user_train, relation_train1, tail_train1 = get_train_instances(
        train_set, relation_set, tail_set)

    item_test = range(ITEM_NUM)
    relation_test, tail_test = [], []

    for i in item_test:
        relation_test.append(relation_set[i])
        tail_test.append(tail_set[i])

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = JNSKR(n_users, n_items, n_relations, n_entities,max_user_pi,max_relation_pi, relation_test, tail_test,negative_c, negative_ck, args)
            model._build_graph()
            optimizer1 = tf.train.AdagradOptimizer(learning_rate=0.05, initial_accumulator_value=1e-8).minimize(
                model.loss)

            sess.run(tf.global_variables_initializer())

            pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []

            cur_best_pre_0=0
            stopping_step=0
            should_stop = False

            for epoch in range(epochs):
                print epoch, "bridge entities training"

                start_t = _writeline_and_time('\tUpdating...')
                shuffle_indices = np.random.permutation(np.arange(len(item_train)))
                item_train_shuffled = item_train[shuffle_indices]
                user_train_shuffled = user_train[shuffle_indices]
                relation_train1_shuffled = relation_train1[shuffle_indices]
                tail_train1_shuffled = tail_train1[shuffle_indices]
                ll = int(len(item_train_shuffled) / batch_size)
                #bridge items
                loss = [0.0, 0.0, 0.0]
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(item_train_shuffled))
                    u_batch = user_train_shuffled[start_index:end_index]
                    i_batch = item_train_shuffled[start_index:end_index]
                    r_batch = relation_train1_shuffled[start_index:end_index]
                    t_batch = tail_train1_shuffled[start_index:end_index]

                    loss0, loss1,loss2,w = train_step2(u_batch, i_batch, r_batch, t_batch,args)
                    loss[0] += loss0
                    loss[1] += loss1
                    loss[2] += loss2
                print w[5][0],w[5][1],w[5][2]

                print('\r\tUpdating: time=%.2f'
                      % (time.time() - start_t))
                print 'loss,loss_1,loss_2 ', loss[0] / ll, loss[1] / ll, loss[2] / ll


                if epoch < epochs:
                    if epoch % args.verbose == 0:
                        users_to_test = list(data_generator.test_user_dict.keys())
                        ret = test(sess, model, users_to_test, item_test)

                        rec_loger.append(ret['recall'])
                        pre_loger.append(ret['precision'])
                        ndcg_loger.append(ret['ndcg'])
                        hit_loger.append(ret['hit_ratio'])

                        final_perf = "recall=[%s], ndcg=[%s]" % \
                                     ('\t'.join(['%.5f' % r for r in ret['recall']]),
                                      '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                        f1.write(final_perf + '\n')
                        f1.flush()
                        print(final_perf)



                        if args.sparsity==1:
                            print "sparsity"
                            users_to_test_list, split_state = data_generator.get_sparsity_split()

                            for i, users_to_test in enumerate(users_to_test_list):
                                ret = test(sess, model, users_to_test,  item_test)
                                final_perf = "recall=[%s], ndcg=[%s]" % \
                                             ('\t'.join(['%.5f' % r for r in ret['recall']]),
                                              '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                                print(final_perf)

                        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)
                        if should_stop == True:
                            break

            recs = np.array(rec_loger)
            pres = np.array(pre_loger)
            ndcgs = np.array(ndcg_loger)
            hit = np.array(hit_loger)

            best_rec_0 = max(recs[:, 0])
            idx = list(recs[:, 0]).index(best_rec_0)

            final_perf = "Best Iter=[%d]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
            print(final_perf)























