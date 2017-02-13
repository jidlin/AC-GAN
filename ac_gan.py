import os

import tensorflow as tf
import numpy as np

import ops

from tfrecords_reader import TFRecordsReader

tf.app.flags.DEFINE_integer('input_height', 64, 'input image height')
tf.app.flags.DEFINE_integer('input_width', 64, 'input image width')
tf.app.flags.DEFINE_integer('input_channels', 1, 'image channels')
tf.app.flags.DEFINE_integer('output_height', 64, 'output image height')
tf.app.flags.DEFINE_integer('output_width', 64, 'output image width')
tf.app.flags.DEFINE_integer('z_dim', 100, 'generator input dim')
tf.app.flags.DEFINE_integer('n_classes', 10, 'number of classes')
tf.app.flags.DEFINE_boolean('crop', True, 'crop image or not')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'learning rate')
tf.app.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam')

FLAGS = tf.app.flags.FLAGS


def inference(images, labels, z):
    generated_images = generator(z, labels)
    source_logits_real, class_logits_real = discriminator(images, labels)
    source_logits_fake, class_logits_fake = discriminator(
        generated_images, labels, reuse=True)

    return [
        source_logits_real, class_logits_real, source_logits_fake,
        class_logits_fake, generated_images
    ]


def loss(labels, source_logits_real, class_logits_real, source_logits_fake,
         class_logits_fake, generated_images):

    labels_one_hot = tf.one_hot(labels, FLAGS.n_classes)

    source_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            source_logits_real, tf.ones_like(source_logits_real)))

    source_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            source_logits_fake, tf.zeros_like(source_logits_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            source_logits_fake, tf.ones_like(source_logits_fake)))

    class_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(class_logits_real,
                                                labels_one_hot))
    class_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(class_logits_fake,
                                                labels_one_hot))

    d_loss = source_loss_real + source_loss_fake + class_loss_real + class_loss_fake

    g_loss =  g_loss + class_loss_real + class_loss_fake

    return d_loss, g_loss


def train(d_loss, g_loss):
    # variables for discriminator
    d_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # variables for generator
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    # train discriminator
    d_optimzer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    train_d_op = d_optimzer.minimize(d_loss, var_list=d_vars)

    # train generator
    g_optimzer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    train_g_op = g_optimzer.minimize(g_loss, var_list=g_vars)

    return train_d_op, train_g_op


def discriminator(images, labels, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # conv1
        conv1 = ops.conv_2d(images, 64, scope="conv1")

        # leakly ReLu
        h1 = ops.leaky_relu(conv1)

        # conv2
        conv2 = ops.conv_2d(h1, 128, scope="conv2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=True)

        # leaky ReLU
        h2 = ops.leaky_relu(norm2)

        # conv3
        conv3 = ops.conv_2d(h2, 256, scope="conv3")

        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=True)

        # leaky ReLU
        h3 = ops.leaky_relu(norm3)

        # conv4
        conv4 = ops.conv_2d(h3, 512, scope="conv4")

        # batch norm
        norm4 = ops.batch_norm(conv4, scope="batch_norm4", is_training=True)

        # leaky ReLU
        h4 = ops.leaky_relu(norm4)

        # reshape
        h4_reshape = tf.reshape(h4, [FLAGS.batch_size, -1])

        # source logits
        source_logits = ops.fc(h4_reshape, 1, scope="source_logits")

        # class logits
        class_logits = ops.fc(
            h4_reshape, FLAGS.n_classes, scope="class_logits")

        return source_logits, class_logits


def generator(z, labels):
    with tf.variable_scope("generator") as scope:
        # labels to one_hot
        labels_one_hot = tf.one_hot(labels, FLAGS.n_classes)

        # concat z and labels
        z_labels = tf.concat(1, [z, labels_one_hot])

        # project z and reshape
        oh, ow = FLAGS.output_height, FLAGS.output_width

        z_labels_ = ops.fc(z_labels, 512 * oh / 16 * ow / 16, scope="project")
        z_labels_ = tf.reshape(z_labels_, [-1, oh / 16, ow / 16, 512])

        # batch norm
        norm0 = ops.batch_norm(
            z_labels_, scope="batch_norm0", is_training=True)

        # ReLU
        h0 = tf.nn.relu(norm0)

        # conv1
        conv1 = ops.conv2d_transpose(
            h0, [FLAGS.batch_size, oh / 8, ow / 8, 256],
            scope="conv_tranpose1")

        # batch norm
        norm1 = ops.batch_norm(conv1, scope="batch_norm1", is_training=True)

        # ReLU
        h1 = tf.nn.relu(norm1)

        # conv2
        conv2 = ops.conv2d_transpose(
            h1, [FLAGS.batch_size, oh / 4, ow / 4, 128],
            scope="conv_tranpose2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=True)

        # ReLU
        h2 = tf.nn.relu(norm2)

        # conv3
        conv3 = ops.conv2d_transpose(
            h2, [FLAGS.batch_size, oh / 2, ow / 2, 64], scope="conv_tranpose3")

        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=True)

        # ReLU
        h3 = tf.nn.relu(norm3)

        # conv4
        conv4 = ops.conv2d_transpose(
            h3, [FLAGS.batch_size, oh, ow, FLAGS.input_channels],
            scope="conv_tranpose4")

        # tanh
        h4 = tf.nn.tanh(conv4)

    return h4


def inputs(batch_size=64):
    crop = FLAGS.crop
    crop_height, crop_width = FLAGS.input_height, FLAGS.input_width
    resize_height, resize_width = FLAGS.output_height, FLAGS.output_width

    reader = TFRecordsReader(
        image_height=28,
        image_width=28,
        image_channels=1,
        image_format="bmp",
        directory="data/mnist",
        filename_pattern="*.tfrecords",
        crop=False,
        crop_height=crop_height,
        crop_width=crop_width,
        resize=True,
        resize_height=resize_height,
        resize_width=resize_height,
        num_examples_per_epoch=64)

    images, labels = reader.inputs(batch_size=64)
    float_images = tf.cast(images, tf.float32)
    float_images = float_images / 127.5 - 1.0

    return float_images, labels
