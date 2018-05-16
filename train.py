import os
import json
import time
import datetime
import logging
import data_utils
import numpy as np
import tensorflow as tf
from model import TextCNN
from sklearn.model_selection import train_test_split

# load parameter file
parameter_file = 'parameters.json'
params = json.loads(open(parameter_file).read())

timestr = time.strftime("%d%m-%H%M")

log_file = params["log_file"] + "_" + params["output_dir"] + "_" + timestr + '.txt'

if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        pass

logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w', format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def train_cnn():
    """Step 0: load sentences, labels, and training parameters"""

    create_test = params["create_test"]

    # load train, cat and and other path configurations from parameter file
    train_file = params['train_file']
    # cat_file = params['cat_file']
    test_set_dir = params['test_set_dir']
    desc_col = params["desc_col"]
    alphabet = params["alphabet"]
    char_seq_len = params["char_seq_len"]
    # dev_set_dir = params['dev_set_dir']

    x_raw, y_raw, labels = data_utils.load_tagged_data(train_file, desc_col, alphabet,
                                                       char_seq_len, ispickle=False)

    max_trans_length = params['char_seq_len']
    logger.debug('The maximum character length set for all transactions: {}'.format(max_trans_length))

    x = np.array(x_raw)
    y = np.array(y_raw)

    if create_test:
        """Step 2: split the original dataset into train and test sets"""
        logger.info("preparing test set")
        x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
        logger.info("saving test set")
        x_test.to_csv(os.path.join(test_set_dir, 'x_test.csv'), index=None)
        y_test.to_csv(os.path.join(test_set_dir, 'y_test.csv'), index=None)

        logger.debug("x_test: {}, y_test: {}".format(len(x_test), len(y_test)))

    else:
        x_ = x
        y_ = y

    logger.info("preparing dev set")

    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, stratify=y_shuffled,
                                                      test_size=params['val_set_ratio'], random_state=42)

    # x_dev.to_csv(os.path.join(dev_set_dir,'x_test.csv'),index=None)
    # x_dev.to_csv(os.path.join(dev_set_dir,'y_test.csv'),index=None)

    """Step 4: save the labels into labels.json since predict.py needs it"""
    logger.info("saving labels into json file")
    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    logger.debug('x_train: {}, x_dev: {}'.format(len(x_train), len(x_dev)))
    logger.debug('y_train: {}, y_dev: {}'.format(len(y_train), len(y_dev)))

    """Step 5: build a graph and cnn object"""
    logger.info("building tensorflow graph")

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(params["alphabet"]) + 1,
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(params['learning_rate'])
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            timestamp = time.strftime("%m%d-%H%M")
            output_dir = params['output_dir']
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", output_dir, timestamp))

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            with open(out_dir + 'parameters.json','w') as f:
                json.dump(params,f)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, step, summaries, loss, acc = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logger.debug("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                train_summary_writer.add_summary(summaries, step)

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, acc, num_correct = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct], feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                # logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                # if writer:
                #    writer.add_summary(summaries, step)
                dev_summary_writer.add_summary(summaries, step)

                return num_correct

            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_utils.batch_iter(x_train, y_train, params['batch_size'],
                                                  params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = data_utils.batch_iter(x_dev, y_dev, params['dev_batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct

                    #total_dev_correct = dev_step(x_dev, y_dev)

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logger.info('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: save the model if it is the best based on accuracy of the dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logger.info('Saved model {} at step {}'.format(path, best_at_step))
                        logger.info('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

            if create_test:
                """Step 7: predict x_test (batch by batch)"""
                test_batches = data_utils.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
                total_test_correct = 0
                for test_batch in test_batches:
                    x_test_batch, y_test_batch = zip(*test_batch)
                    num_test_correct = dev_step(x_test_batch, y_test_batch)
                    total_test_correct += num_test_correct

                total_test_correct = dev_step(x_test, y_test)

                test_accuracy = float(total_test_correct) / len(y_test)
                print('Accuracy on test set is {}'.format(test_accuracy))
                print('The training is complete')


if __name__ == '__main__':
    train_cnn()

