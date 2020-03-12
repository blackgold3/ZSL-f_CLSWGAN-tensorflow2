import tensorflow as tf
import numpy as np

def data_load(opt):
    datapath = opt.path + opt.dataset
    att = np.loadtxt(datapath + '/general/att.csv', delimiter=',')
    features = np.loadtxt(datapath +"/general/features.csv", delimiter=',')
    labels = np.loadtxt(datapath +"/general/labels.csv", delimiter=',')
    test_seen_loc = np.loadtxt(datapath +"/ps_split/test_seen_loc.csv", delimiter=',')
    test_unseen_loc = np.loadtxt(datapath +"/ps_split/test_unseen_loc.csv", delimiter=',')
    trainval_loc = np.loadtxt(datapath +"/ps_split/trainval_loc.csv", delimiter=',')
    # train_loc = np.loadtxt(datapath +"/ps_split/train_loc.csv", delimiter=',')
    # val_loc = np.loadtxt(datapath + "/ps_split/val_loc.csv", delimiter=',')

    # 特征 #
    x_trainval = features[:, trainval_loc.astype(int) - 1]
    test_seen = features[:, test_seen_loc.astype(int) - 1]
    test_unseen = features[:, test_unseen_loc.astype(int) - 1]
    # x_train = features[:, train_loc.astype(int) - 1]
    # x_val = features[:, val_loc.astype(int) - 1]

    # 标签 #
    y_trainval = labels[trainval_loc.astype(int) - 1]
    y_seen = labels[test_seen_loc.astype(int) - 1]
    y_unseen = labels[test_unseen_loc.astype(int) - 1]
    # y_train = labels[train_loc.astype(int) - 1]
    # y_val = labels[val_loc.astype(int) - 1]

    # 返回 #
    x_trainval = tf.convert_to_tensor(x_trainval, dtype=tf.float32)
    y_trainval = tf.convert_to_tensor(y_trainval, dtype=tf.int32)
    # x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    # y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    # x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    # y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    test_unseen = tf.convert_to_tensor(test_unseen, dtype=tf.float32)
    y_unseen = tf.convert_to_tensor(y_unseen, dtype=tf.int32)
    test_seen = tf.convert_to_tensor(test_seen, dtype=tf.float32)
    y_seen = tf.convert_to_tensor(y_seen, dtype=tf.int32)
    att = tf.convert_to_tensor(att, dtype=tf.float32)


    trainval = [x_trainval, y_trainval]
    unseen = [test_unseen, y_unseen]
    seen = [test_seen, y_seen]
    # train = [x_train, y_train]
    # val = [x_val, y_val]
    return att, trainval, unseen, seen






