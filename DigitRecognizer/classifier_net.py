import tensorflow as tf


layer = {}

def swift(X):
    return X*tf.nn.sigmoid(X)

def basic_conv_block(X, outchannel, name):
    conv1 = tf.layers.conv2d(X, outchannel, kernel_size=[3,3], padding="same", name=name)
    layer[name] = conv1
    bn1 = tf.layers.batch_normalization(conv1)
    activate1 = swift(bn1)
    return activate1

def inception_conv_block(X, channel, name):
    cos = [channel/4] * 4
    conv_1 = tf.layers.conv2d(X, cos[0], kernel_size=(1,1), strides=(2,2), padding="same", name=name+"_conv1")
    layer[name+"_conv1"] = conv_1

    conv_2 = tf.layers.conv2d(X, 2*cos[1], kernel_size=(1,1), strides=(2,2), padding="same")
    bn_2 = tf.layers.batch_normalization(conv_2)
    ac_2 = swift(bn_2)
    conv_2 = tf.layers.conv2d(ac_2, cos[1], kernel_size=(3,3), padding="same", name=name+"_conv2")
    layer[name + "_conv2"] = conv_2

    conv_3 = tf.layers.conv2d(X, 2 * cos[2], kernel_size=(1, 1), strides=(2, 2), padding="same")
    bn_3 = tf.layers.batch_normalization(conv_3)
    ac_3 = swift(bn_3)
    conv_3 = tf.layers.conv2d(ac_3, cos[2], kernel_size=(5,5), padding="same", name=name+"_conv3")
    layer[name + "_conv3"] = conv_3

    conv_4 = tf.layers.max_pooling2d(X, pool_size=(3,3), strides=(2,2), padding="same")
    conv_4 = tf.layers.conv2d(conv_4, cos[3], kernel_size=(1,1), padding="same", name=name+"_conv4")
    layer[name + "_conv4"] = conv_4

    concat = tf.concat([conv_1, conv_2, conv_3, conv_4], axis=-1)
    return concat



def build_network(X):
    with tf.variable_scope('global/network_scope'):
        conv1 = basic_conv_block(X, 128, name = "conv1")
        iBlock1 = inception_conv_block(conv1, 128, name = "conv2")
        iBlock2 = inception_conv_block(iBlock1, 256, name = "inception_block")
        conv3 = basic_conv_block(iBlock2, 256, name = "conv3")
        conv4 = basic_conv_block(conv3, 512, name = "conv4")
        size = conv4.get_shape().as_list()
        assert(len(size) == 4)
        flat = tf.reshape(conv4, shape=(-1, size[1]*size[2]*size[3]))
        dense1 = tf.layers.dense(inputs=flat, units=1024)
        dense1 = swift(dense1)
        drop1 = tf.layers.dropout(dense1, rate=0.2)

        dense2 = tf.layers.dense(inputs=drop1, units=512)
        dense2 = swift(dense2)
        drop2 = tf.layers.dropout(dense2, rate=0.2)

        dense3 = tf.layers.dense(inputs=drop2, units=10, name="logits")
        return dense3


def build_network(X):
    net = build_network(X)
    return net, layer