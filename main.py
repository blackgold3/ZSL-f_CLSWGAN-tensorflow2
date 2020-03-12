import read_data
import WGAN
import argparse
import tensorflow as tf

# 参数设置 #
parser = argparse.ArgumentParser()
# data load
parser.add_argument('--dataset', default='AWA2', help='dataset')
parser.add_argument('--path', default='data/', help='dataset')
# choose
parser.add_argument('--pre_classifier_read', type=bool, default=False, help='if read pre_train classifier weights')
# pre_train model setting
parser.add_argument('--pre_class_lr', type=float, default=0.0001, help='class learing rate')
parser.add_argument('--class_batch', type=int, default=100, help='class batch')
parser.add_argument('--pre_epoch', type=int, default=50, help='pre_train cls epoch')
# train setting
parser.add_argument('--gp_lambda', type=float, default=10, help= "GP's weight for WGAN-GP")
parser.add_argument('--cls_beita', type=float, default=0.01, help= "cps's weight for generate training")
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--train_lr', type=float, default=0.0001, help= "G-D train learning rate")
parser.add_argument('--train_batch', type=int, default=64, help='train batch')
parser.add_argument('--calibrate', type=float, default=0.95, help='calibrated factor, 0 means normal GZSL, 1 means all check unseen class')
parser.add_argument('--valid_epoch', type=int, default=10, help='valid epoch in every train step')
# generate data setting
parser.add_argument('--generate_num', type=int, default=1000, help='number features to generate per class')
# classifier for test setting
parser.add_argument('--class_lr', type=float, default=0.0001, help='class learing rate')
parser.add_argument('--class_regularizer', type=float, default=0.001, help='class regularizer')
# tensorflow setting
parser.add_argument('--random_seed', type=int, help='generate a random seed')
parser.add_argument('--device', type=int, default=0, help='choose a device')
opt = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[opt.device], True)
tf.config.experimental.set_visible_devices(gpus[opt.device], 'GPU')

opt.random_seed = int(tf.squeeze(tf.random.uniform(shape=(1,), minval= 1, maxval=10000, dtype=tf.int64)))
print("随机种子:", opt.random_seed)
print(opt)
# 参数设置 #

# 读取数据 #
att, trainval, unseen, seen= read_data.data_load(opt)
# 读取数据 #

# 计算准确度 #
(max_zsl, max_gu, max_gs, max_h) = WGAN.wgan(att, trainval, unseen, seen, opt)


file = open('log.csv', 'w')
log = []
log.append(['max_zsl', float(max_zsl)])
log.append(['max_gu', float(max_gu)])
log.append(['max_gs', float(max_gs)])
log.append(['max_h', float(max_h)])
log.append(['seed', float(opt.random_seed)])

writer = csv.writer(file)
for i in range(len(log)):
    writer.writerow(log[i])


print("max-zsl")
print('\tZSL->%f'%max_zsl)
print("max-gzsl:")
print('\tUnseen->%f'%max_gu)
print("\tSeen->%f"%max_gs)
print("\tH->%f"%max_h) 
