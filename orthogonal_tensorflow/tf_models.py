import numpy as np
import argparse
import tensorflow as tf

from EUNN import EUNNCell

import os

def main(model, n_epochs, n_batch, n_hidden, capacity, comp, fft):

    # --- Set data params ----------------
    n_input = 24
    n_classes = 8


    # --- Create graph and compute gradients ----------------------
    x = tf.placeholder("float", [None, 1724, n_input])
    y = tf.placeholder("int64", [None, 1724, n_classes])
    

    # --- Input to hidden layer ----------------------
    if model == "LSTM":
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
        hidden_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32)
    elif model == "EUNN":
        with tf.variable_scope("cell_fw"):
            cell_fw = EUNNCell(n_hidden, capacity, fft, comp) #requires comp = False
        with tf.variable_scope("cell_bw"):
            cell_bw = EUNNCell(n_hidden, capacity, fft, comp)
        hidden_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32)

    hidden_out = tf.reduce_mean(tf.reshape(hidden_out, [-1, 2, 1724, n_hidden]), axis=1)

    # --- Hidden Layer to Output ----------------------
    V_init_val = np.sqrt(6.)/np.sqrt(n_classes + n_input)

    V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
            dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[n_classes], \
            dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    temp_out = tf.tensordot(hidden_out, V_weights, [[-1], [0]])
    output_data = tf.nn.bias_add(temp_out, V_bias) 
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_data, labels=y))
    correct_pred = tf.equal(tf.argmax(output_data, axis=-1), tf.argmax(y, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # Print Number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


    # --- Initialization --------------------------------------------------
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9).minimize(cost)
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)
    init = tf.global_variables_initializer()

    # --- Training Loop ---------------------------------------------------------------


    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.log_device_placement = False
    config.allow_soft_placement = False
    with tf.Session(config=config) as sess:

        # --- Create data --------------------

        data_dir = "onehot_q8_mono" 
        
        train = [np.load(os.path.join(data_dir, 'x_train.npy')), 
                    np.load(os.path.join(data_dir, 'y_train.npy'))]
        val = [np.load(os.path.join(data_dir, 'x_val.npy')),
                    np.load(os.path.join(data_dir, 'y_val.npy'))]
        test = [np.load(os.path.join(data_dir, 'x_test.npy')),
                    np.load(os.path.join(data_dir, 'y_test.npy'))]

        train_size = train[0].shape[0]
        val_size = val[0].shape[0]
        test_size = test[0].shape[0]


        sess.run(init)

        epoch = 0
        while epoch < n_epochs:

            batch_index = 0
            num_batches = int(np.ceil(train_size/n_batch))
            for i in range(num_batches):
                batch_x = train[0][batch_index: min(batch_index + n_batch, train_size)]
                batch_y = train[1][batch_index: min(batch_index + n_batch, train_size)]
                batch_index += n_batch

                loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y})

                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                print(" Epoch: " + str(epoch) + " Batch: " + str(i) + "/" + str(num_batches) + \
                    ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))

            val_acc_list = []
            val_loss_list = []
            batch_index = 0
            for i in range(int(np.ceil(val_size/n_batch))):
                batch_x = val[0][batch_index: min(batch_index + n_batch, val_size)]
                batch_y = val[1][batch_index: min(batch_index + n_batch, val_size)]
                batch_index += n_batch

                val_acc_list.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
                val_loss_list.append(sess.run(cost, feed_dict={x: batch_x, y: batch_y}))

            val_acc = np.mean(val_acc_list)
            val_loss = np.mean(val_loss_list)
            print("Epoch: " + str(epoch) + ", Validation Loss= " + \
                "{:.6f}".format(val_loss) + ", Validation Accuracy= " + \
                "{:.5f}".format(val_acc))

            epoch += 1
                

        print("Optimization Finished!")

        
        # --- test ----------------------

        test_acc_list = []
        test_loss_list = []

        batch_index = 0
        for i in range(int(np.ceil(test_size/n_batch))):
            batch_x = test[0][batch_index: min(batch_index + n_batch, test_size)]
            batch_y = test[1][batch_index: min(batch_index + n_batch, test_size)]
            batch_index += n_batch

            test_acc_list.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
            test_loss_list.append(sess.run(cost, feed_dict={x: batch_x, y: batch_y}))

        test_acc = np.mean(test_acc_list)
        test_loss = np.mean(test_loss_list)

        print("Test result: Loss= " + "{:.6f}".format(test_loss) + ", Accuracy= " + "{:.5f}".format(test_acc))


                



if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Efficient Unitary Recurrent Neural Network")
    parser.add_argument("model", default='LSTM', help='Model name: LSTM, EUNN')
    parser.add_argument('--n_epochs', '-I', type=int, default=10, help='number of epochs')
    parser.add_argument('--n_batch', '-B', type=int, default=32, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int, default=128, help='hidden layer size')
    parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is True: complex domain')
    parser.add_argument('--fft', '-F', type=str, default="False", help='fft style, only for EUNN, default is False: tunable style')

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
                'model': dict['model'],
                'n_epochs': dict['n_epochs'],
                'n_batch': dict['n_batch'],
                'n_hidden': dict['n_hidden'],
                'capacity': dict['capacity'],
                'comp': dict['comp'],
                'fft': dict['fft'],
            }


    main(**kwargs)