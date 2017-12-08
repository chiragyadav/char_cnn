# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
import math


class CharCNN:
    """
    A CNN for text classification.
    based on the Character-level Convolutional Networks for Text Classification paper.
    """

    def __init__(self, num_classes=42, filter_widths=(7, 7, 3, 3, 3, 3),
                 num_filters_per_layer=(100, 200, 300, 400, 500, 600), l2_reg_lambda=0.0,
                 seq_max_length=110, char_voc_size=70):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, char_voc_size, seq_max_length, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # ================ Layer 1 ================

        # conv2d: takes input: [batch, in_height, in_width, in_channels] and
        # filter: [filter_height, filter_width, in_channels, out_channels]

        with tf.name_scope("conv-maxpool-1"):
            filter_shape = [char_voc_size, filter_widths[0], 1, num_filters_per_layer[0]]

            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_layer[0]]), name="b")

            ''' set stride for first dimension to char_voc_size, so we can still use valid padding. 
            This way input a tweet as [voc_size, max_seq] and we return an output [1, max_seq].
            In original, there is no padding and the output is [1, max_seq-7]
            '''

            conv = tf.nn.conv2d(self.input_x, W, strides=[1, char_voc_size, 1, 1], padding="SAME", name="conv1")
            # print(conv.get_shape())
            # conv shape : [batch, voc, seq_max, filters]
            non_lin_feature_map = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                non_lin_feature_map,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='SAME',
                name="pool1")

            # output: pooled, shape: [batch, 1, ceil(seq_max/3), num_filters_per_layer[0]]

        # todo: apply pooling here too (or not)
        # ================ Layer 2 ================
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [1, filter_widths[1], num_filters_per_layer[0], num_filters_per_layer[1]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_layer[1]]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            non_lin_feature_map = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # pooled = tf.nn.max_pool(
            #     h,
            #     ksize=[1, 1, 3, 1],
            #     strides=[1, 1, 3, 1],
            #     padding='VALID',
            #     name="pool2")

            # print(non_lin_feature_map.get_shape())
            # output: non_lin_feature_map, shape: [batch, 1, ceil(seq_max/3), num_filters_per_layer[1]]

        # ================ Layer 3 ================
        with tf.name_scope("conv-3"):
            filter_shape = [1, filter_widths[2], num_filters_per_layer[1], num_filters_per_layer[2]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_layer[2]]), name="b")
            conv = tf.nn.conv2d(non_lin_feature_map, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            non_lin_feature_map = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # output: non_lin_feature_map, shape: [batch, 1, ceil(seq_max/3), num_filters_per_layer[2]]

        # ================ Layer 4 ================
        with tf.name_scope("conv-4"):
            filter_shape = [1, filter_widths[3], num_filters_per_layer[2], num_filters_per_layer[3]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_layer[3]]), name="b")
            conv = tf.nn.conv2d(non_lin_feature_map, W, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
            non_lin_feature_map = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # output: non_lin_feature_map, shape: [batch, 1, ceil(seq_max/3), num_filters_per_layer[3]]

        # ================ Layer 5 ================
        with tf.name_scope("conv-5"):
            filter_shape = [1, filter_widths[4], num_filters_per_layer[3], num_filters_per_layer[4]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_layer[4]]), name="b")
            conv = tf.nn.conv2d(non_lin_feature_map, W, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
            non_lin_feature_map = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # output: non_lin_feature_map, shape: [batch, 1, ceil(seq_max/3), num_filters_per_layer[4]]

        # ================ Layer 6 ================
        with tf.name_scope("conv-maxpool-6"):
            filter_shape = [1, filter_widths[5], num_filters_per_layer[4], num_filters_per_layer[5]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_layer[5]]), name="b")
            conv = tf.nn.conv2d(non_lin_feature_map, W, strides=[1, 1, 1, 1], padding="SAME", name="conv6")
            non_lin_feature_map = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                non_lin_feature_map,
                ksize=[1, 1, 3, 1],
                strides=[1, 1, 3, 1],
                padding='SAME',
                name="pool6")

            # output: pooled, shape: [batch, 1, ceil(seq_max/9), num_filters_per_layer[5]]

        # ================ Layer 7 ================
        '''
        the length of the output depends on how many pooling operations we did. In this case, we used 2 poolings with
        a pooling size of 3, so the final length is the seq_max/9. Since we use padding, it's ceil(seq_max/9)
        '''
        num_features_total = math.ceil(seq_max_length/9) * num_filters_per_layer[5]
        # reshape with -1 keeps that dimension constant while flattening out the rest of the shape (?)
        h_pool_flat = tf.reshape(pooled, [-1, num_features_total])

        # Add dropout
        with tf.name_scope("dropout-1"):
            drop1 = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Fully connected layer 1
        # todo: why is 1024 hardcoded? should be adjustable (in paper: output large/small = 2048/1024)
        with tf.name_scope("fc-1"):
            W = tf.Variable(tf.truncated_normal([num_features_total, 1024], stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[num_features_total, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")

        # ================ Layer 8 ================
        # Add dropout
        with tf.name_scope("dropout-2"):
            drop2 = tf.nn.dropout(fc_1_output, self.dropout_keep_prob)

        # Fully connected layer 2
        with tf.name_scope("fc-2"):
            W = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[1024, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

        # ================ Layer 9 ================
        # Fully connected layer 3
        with tf.name_scope("fc-3"):
            W = tf.Variable(tf.truncated_normal([1024, num_classes], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")

            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # ================ Loss and Accuracy ================

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # todo: correct loss calculation? -> check PTB project and copy loss from there, see if same result
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')