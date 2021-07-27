import tensorflow as tf
import numpy as np
import math
import numpy.linalg as la

from model.tgcn import tgcnCell
from preprocess.adjacent import get_adjacent_matrix
from preprocess.feature import Matrix
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

time_start = time.time()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 5000, 'Number of epochs to train.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
model_name = FLAGS.model_name
train_rate = FLAGS.train_rate
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = 32

len_clossness = 5
len_period = 3
len_trend = 1
len_test = 200
adj = get_adjacent_matrix()
X_train, Y_train, X_test, Y_test, mmn, timestamp_train, timestamp_test = Matrix(
    len_closeness=len_clossness, len_period=len_period, len_trend=len_trend,
    len_test=len_test)

x_data = X_train[0]
y_data = Y_train
time_len = x_data.shape[0]
num_nodes = x_data.shape[2]
totalbatch = int(X_train[0].shape[0] / batch_size)
training_data_count = len(X_train)
max_value = 1171
a = X_test[0][0:100, :]
b = X_test[1][0:100, :]
c = X_test[2][0:100, :]

d = Y_test[0:100]
print()

def model(inputs_c, inputs_p, inputs_t, _weights, _biases, external_dim):
    final_output = []
    if inputs_c is not None:
        cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
        _X = tf.unstack(inputs_c, axis=1)
        outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
        m = []
        for i in outputs:
            o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
            o = tf.reshape(o, shape=[-1, gru_units])
            m.append(o)
        last_output = m[-1]
        output = tf.matmul(last_output, _weights['out']) + _biases['out']
        output = tf.reshape(output, shape=[-1, num_nodes, 1])
        output = tf.transpose(output, perm=[0, 1, 2])
        output = tf.reshape(output, shape=[-1, num_nodes])
        final_output.append(output)
    if inputs_p is not None:
        cell_3 = tgcnCell(gru_units, adj, num_nodes=num_nodes, index=2)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell_3], state_is_tuple=True)
        _X = tf.unstack(inputs_p, axis=1)
        outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
        m = []
        for i in outputs:
            o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
            o = tf.reshape(o, shape=[-1, gru_units])
            m.append(o)
        last_output = m[-1]
        output = tf.matmul(last_output, _weights['out']) + _biases['out']
        output = tf.reshape(output, shape=[-1, num_nodes, 1])
        output = tf.transpose(output, perm=[0, 1, 2])
        output = tf.reshape(output, shape=[-1, num_nodes])
        final_output.append(output)
    if inputs_t is not None:
        cell_3 = tgcnCell(gru_units, adj, num_nodes=num_nodes, index=3)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell_3], state_is_tuple=True)
        _X = tf.unstack(inputs_t, axis=1)
        outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
        m = []
        for i in outputs:
            o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
            o = tf.reshape(o, shape=[-1, gru_units])
            m.append(o)
        last_output = m[-1]
        output = tf.matmul(last_output, _weights['out']) + _biases['out']
        output = tf.reshape(output, shape=[-1, num_nodes, 1])
        output = tf.transpose(output, perm=[0, 1, 2])
        output = tf.reshape(output, shape=[-1, num_nodes])
        final_output.append(output)

    if len(final_output) == 1:
        return tf.tanh(final_output[0])
    else:
        final = final_output[0]
        for i in range(1, len(final_output)):
            final = final + final_output[i]
            return tf.tanh(final)


inputs = tf.placeholder(tf.float32, shape=[None, 3, num_nodes])
inputs_c = tf.placeholder(tf.float32, shape=[None, len_clossness, num_nodes], name="inputs_c")
inputs_p = tf.placeholder(tf.float32, shape=[None, len_period, num_nodes], name="inputs_p")
inputs_t = t = tf.placeholder(tf.float32, shape=[None, len_trend, num_nodes], name="inputs_t")
# external = tf.placeholder(tf.float32, shape=[None, 1], name="external")
labels = tf.placeholder(tf.float32, shape=[None, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, 1], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([1]), name='bias_o')}

pred = model(inputs_c, inputs_p, inputs_t, weights, biases, None)
y_pred = pred
tf.add_to_collection("y_predict", y_pred)

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
mae_c = tf.reduce_mean(abs(y_pred - label))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
# sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = './out/'
# out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_lr%r_batch%r_unit%r_close%r_period%r_trend%r_epoch%r' % (
    model_name, lr, batch_size, gru_units, len_clossness, len_period, len_trend, training_epoch)
path = out + path1


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var


x_axe, batch_loss, batch_rmse, batch_mae, batch_pred = [], [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch_c = X_train[0][m * batch_size: (m + 1) * batch_size]
        mini_batch_p = X_train[1][m * batch_size: (m + 1) * batch_size]
        mini_batch_t = X_train[2][m * batch_size: (m + 1) * batch_size]
        # mini_batch_e = X_train[3][m * batch_size: (m + 1) * batch_size]

        mini_label = Y_train[m * batch_size: (m + 1) * batch_size]

        _, loss1, rmse1, mae1, train_output = sess.run([optimizer, loss, error, mae_c, y_pred],
                                                       feed_dict={inputs_c: mini_batch_c, inputs_p: mini_batch_p,
                                                                  inputs_t: mini_batch_t,
                                                                  # external: mini_batch_e,
                                                                  labels: mini_label})
        # print(rmse1)
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)
        batch_mae.append(mae1 * max_value)

    # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict={inputs_c: X_test[0][0:100, :], inputs_p: X_test[1][0:100, :],
                                                    inputs_t: X_test[2][0:100, :],
                                                    # external: X_test[3],
                                                    labels: Y_test[0:100]})
    test_label = np.reshape(Y_test[0:100], [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'train_mae:{:.4}'.format(batch_mae[-1]),
          'train_loss:{:.4}'.format(batch_loss[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse * max_value),
          'test_mae:{:.4}'.format(mae * max_value),
          'test_acc:{:.4}'.format(acc))
    if (epoch % 100 == 0):
        saver.save(sess, './out' + '/model_lr%r_closeness%r_period%r_trend%r/model_%r' % (
            lr, len_clossness, len_period, len_trend, epoch),
                   global_step=epoch)

time_end = time.time()
print(time_end - time_start, 's')
