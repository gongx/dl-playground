import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
from DataLoader import DataLoader
import shutil
from configuration import cfg
from classifier_net import build_network

def split_train_validate_set(data_set_path, split_ratio=0.2):
    # read the csv to pandas df
    data_set = pd.read_csv(data_set_path)

    # random shuffle to split train_set and validate_set
    split_mask = np.random.rand(len(data_set)) < (1 - split_ratio)
    train_set = data_set[split_mask]
    validate_set = data_set[~split_mask]

    # Convert train_set/validata_set to numpy ndarray
    train_set_np = train_set.as_matrix()
    train_labels = train_set_np[:, 0].astype('float32')
    train_data = train_set_np[:, 1:].astype('float32')
    train_data = train_data/255
    train_data = train_data.reshape(len(train_data), 28, 28)
    train_data = np.expand_dims(train_data, axis=-1)
    train_labels = pd.get_dummies(train_labels).as_matrix()


    validate_set_np = validate_set.as_matrix()
    validate_labels = validate_set_np[:, 0].astype('float32')
    validate_data = validate_set_np[:, 1:].astype('float32')
    validate_data = validate_data.reshape(len(validate_data), 28, 28)
    validate_data /= 255
    validate_data = np.expand_dims(validate_data, axis=-1)
    validate_labels = pd.get_dummies(validate_labels).as_matrix()

    return train_data, train_labels, validate_data, validate_labels

def train_network(sess, clear=True, continue_training=False):
    if not os.path.exists(cfg.DIR.tensorboard_log_dir):
        os.mkdir(cfg.DIR.tensorboard_log_dir)
    else:
        if clear:
            shutil.rmtree(cfg.DIR.tensorboard_log_dir)

    if not os.path.exists(cfg.DIR.model_save_dir):
        os.mkdir(cfg.DIR.model_save_dir)

    writer = tf.summary.FileWriter(cfg.DIR.tensorboard_log_dir)
    writer.add_graph(sess.graph)

    # initialize the global parameters
    sess.run(tf.global_variables_initializer())

    if continue_training:
        value_list = []
        value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/network_scope'))
        restore = tf.train.Saver(value_list)
        restore.restore(sess, tf.train.latest_checkpoint(cfg.DIR.model_save_dir))


    X = tf.placeholder(shape=(-1, 28, 28, 1), dtype=tf.float32)
    Y = tf.placeholder(shape=(-1, 10), dtype=tf.float32)

    logits, layer = build_network(X)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    train_data, train_labels, validate_data, validate_labels = split_train_validate_set(cfg.DIR.training_data)

    train_data_loader = DataLoader(train_data, train_labels)

    val_data_loader = DataLoader(validate_data, validate_labels)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for key, value in layer.items():
        tf.summary.histogram(key, value)

    tf.summary.scalar("train_loss", loss)
    tf.summary.scalar("train_accuracy", accuracy)

    summary_op = tf.summary.merge_all()

    val_loss_holder = tf.placeholder(tf.float32)
    val_loss_tensor = tf.summary.scalar("val_loss", val_loss_holder)

    val_accuracy_holder = tf.placeholder(tf.float32)
    val_accuracy_tensor = tf.summary.scalar("val_accuracy", val_accuracy_holder)

    saver = tf.train.Saver(max_to_keep=10)

    if not os.path.exists(cfg.DIR.model_save_dir):
        os.makedirs(cfg.DIR.model_save_dir)

    start_time = time.time()

    tf.get_default_graph().finalize()

    train_step = 0
    val_step = 0
    for epoch in range(1, cfg.TRAIN.EPOCHS + 1):
        while(train_data_loader.hasNextBatch()):
            train_step += 1
            batch_data, batch_label = train_data_loader.getNextBatch(cfg.TRAIN.BATCH_SIZE)
            _,_,_,summary = sess.run([loss, accuracy, optimizer, summary_op], feed_dict={X:batch_data, Y:batch_label})
            writer.add_summary(summary, train_step)
        print("Epoch %d finished." % epoch)
        train_data_loader.reset()

        if epoch % cfg.TRAIN.SAVE_STEPS == 0:
            filename = 'digit_recognizer_{:d}'.format(epoch)
            filename = os.path.join(cfg.DIR.model_save_dir , filename)
            saver.save(sess, filename, global_step=epoch)

        if (epoch % cfg.TRAIN.VALIDATE_EPOCHES == 0):
            while val_data_loader.hasNextBatch():
                val_step += 1
                batch_val_data, batch_val_label = val_data_loader.getNextBatch(cfg.TRAIN.BATCH_SIZE)
                val_loss, val_accuracy, _ = sess.run([loss, accuracy],
                                            feed_dict={X: batch_val_data, Y: batch_val_label})

                feed = {val_loss_holder: val_loss}
                val_loss_str = sess.run(val_loss_tensor, feed_dict=feed)
                writer.add_summary(val_loss_str, val_step)

                feed1 = {val_accuracy_holder: val_accuracy}
                val_accuracy_str = sess.run(val_accuracy_tensor, feed_dict=feed1)
                writer.add_summary(val_accuracy_str, val_step)

            val_data_loader.reset()

    filename = os.path.join(cfg.DIR.model_save_dir, 'digit_recognizer_final')
    saver.save(sess, filename)
    end_time = time.time()

    print("The total time used in training: {}".format(end_time - start_time))


def predict(sess):
    # initialize the global parameters
    sess.run(tf.global_variables_initializer())

    value_list = []
    value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global/network_scope'))
    restore = tf.train.Saver(value_list)
    restore.restore(sess, tf.train.latest_checkpoint(cfg.DIR.model_save_dir))

    X = tf.placeholder(shape=(-1, 28, 28, 1), dtype=tf.float32)
    logits, _ = build_network(X)
    predict = tf.argmax(logits, 1)

    test_images = pd.read_csv('../input/test.csv').values
    test_images = test_images.astype(np.float)

    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)

    predicted_lables = np.zeros(test_images.shape[0])

    BATCH_SIZE = cfg.TEST.BATCH_SIZE
    num_of_batches = int(len(test_images)/BATCH_SIZE)
    remains = len(test_images) - num_of_batches * BATCH_SIZE

    for i in range(num_of_batches):
        predicted_lables[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = predict.eval(
            feed_dict={X: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})

    if remains > 0:
        predicted_lables[num_of_batches * BATCH_SIZE:len(test_images)] = predict.eval(
            feed_dict={X: test_images[num_of_batches * BATCH_SIZE:len(test_images)]})

    np.savetxt(os.join(cfg.DIR.data_dir, 'submission_softmax.csv'),
               np.c_[range(1, len(test_images) + 1), predicted_lables],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_network(sess=sess)
        #predict(sess=sess)



