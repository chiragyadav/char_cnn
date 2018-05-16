import pandas as pd
import numpy as np
import pickle
import logging


def load_tagged_data(filename, desc_col, alphabet, char_seq_length, ispickle=True, **kwargs):
    """
    Load the dataframe from either csv or pickle and options to drop duplicates and null values.
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading data file")
    
    if ispickle:
        with open(filename, 'rb') as f:
            df = pickle.load(f, encoding='UTF-8')
            
    else:
        if "sep" in kwargs:
            sep = kwargs["sep"]
        else:
            sep = ','
        df = pd.read_csv(filename, sep=sep)
    
    logger.info("number of rows in file {}".format(df.shape[0]))
    df.columns = [col.lower() for col in df.columns]
    
    # Map the actual labels to one hot labels
    labels = sorted(df['res_category'].unique())
    logger.info("numbers of distinct categories value {}".format(len(labels)))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    # converting labels to one hot encoded values
    y_raw = df['res_category'].apply(lambda y: label_dict[y]).tolist()
    
    x_raw = df[desc_col]
    x_raw = x_raw.apply(lambda x: str(x))
    logging.info("applying padding")
    x_raw = [pad_sentence(x, char_seq_length) for x in x_raw]

    alphabet_to_int = {}

    for i, char in enumerate(alphabet):
        alphabet_to_int[char] = i

    logging.info("converting to int")
    x_raw = [string_to_int8_conversion(x, alphabet_to_int) for x in x_raw]

    #logging.info("converting character sequences to one-hot vectors for cnn input")
    #x_raw = [get_one_hot(x, alphabet, char_seq_length) for x in x_raw]

    logging.info("data loading complete")
    return x_raw, y_raw, labels


def pad_sentence(char_seq, char_seq_length, padding_char=" "):

    if(len(char_seq)) > char_seq_length:
        return char_seq[:char_seq_length]
    else:
        num_padding = char_seq_length - len(char_seq)
        new_char_seq = char_seq + padding_char * num_padding
        return new_char_seq


def string_to_int8_conversion(char_seq, alphabet_to_int):
    x = np.array([alphabet_to_int.get(char, len(alphabet_to_int)) for char in char_seq], dtype=np.int8)
    return x


def get_one_hot(int_char_seq, alphabet, char_seq_length):
    one_hot = np.zeros(shape=[len(alphabet) + 1, char_seq_length, 1])
    for char_index, int_char in enumerate(int_char_seq):
        one_hot[int_char][char_index][0] = 1
    return one_hot


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))

        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = x_shuffled[start_index:end_index], y_shuffled[start_index:end_index]
            batch = list(zip(x_batch, y_batch))
            yield batch
            
def batch_iter_test(x, batch_size):
    """
    Generates a batch iterator for a test dataset.
    """
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_batch = x[start_index:end_index]
        yield x_batch