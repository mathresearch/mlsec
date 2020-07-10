import os
import random
import argparse
import re
import mmh3
import numpy as np
import pickle
import logging

import mxnet as mx

try:
    import Queue as queue
except ImportError:
    import queue

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


def extract_features(string, hash_dim, split_regex=rb"\s+"):
    tokens = re.split(pattern=split_regex, string=string)
    hash_buckets = [(mmh3.hash(w) % hash_dim) for w in tokens]
    buckets, counts = np.unique(hash_buckets, return_counts=True)
    feature_values = np.zeros(hash_dim)
    for bucket, count in zip(buckets, counts):
        feature_values[bucket] = count
    return feature_values


def write_list(path_out, list_file_names):
    """Write the meta-data information
    The format is as below,
    integer_file_index \t float_label_index \t path_to_file
    Note that the blank between number and tab is only used for readability.
    
    Parameters
    ----------
    path_out: string
    list_file_names: list
    """
    with open(path_out, 'w') as fout:
        for i, item in enumerate(list_file_names):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)


def create_list_file(args):
    """
    Read the meta-data information
    :param args: 
    :return: 
    """
    logging.info("Creating a .lst file...")
    benign_folder = args.benign
    mal_folder = args.malicious

    count = 0
    list_file_names = []
    for file in os.listdir(benign_folder):
        file = os.path.join(benign_folder, file)
        list_file_names.append((count, file, 0))
        count += 1

    for file in os.listdir(mal_folder):
        file = os.path.join(mal_folder, file)
        list_file_names.append((count, file, 1))
        count += 1

    if args.shuffle is True:
        random.seed(args.seed)
        random.shuffle(list_file_names)

    if args.train:
        filename = args.output + 'file_train.lst'
    else:
        filename = args.output + 'file_val.lst'
    write_list(filename, list_file_names)


def read_list(path_in):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    Returns
    -------
    item iterator that contains information in .lst file
    """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            if line_len < 3:
                print('lst should have at least has three parts, but only has %s parts for %s' % (
                    line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


def data_encode(args, idx, item, q_out):
    """Reads, preprocesses, packs the data of feature size and put it back in output queue.
    Parameters
    ----------
    args: object
    idx: int
    item: list
    q_out: queue
    """
    filepath = item[1]
    header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    with open(filepath, 'rb') as f:
        content = f.read()
    data = extract_features(content, hash_dim=args.feature_size, split_regex=rb"\s+")
    data = pickle.dumps(data)
    s = mx.recordio.pack(header, data)
    q_out.put((idx, s, item))


def create_recordIO_file(args):
    """
    Generate a RecordIO file
    :param args: 
    :return: 
    """
    logging.info("Creating a .rec and .idx file...")
    q_out = queue.Queue()
    if args.train:
        fname_rec = args.output + 'train.rec'
        fname_idx = args.output + 'train.idx'
        fname_lst = args.output + 'file_train.lst'
    else:
        fname_rec = args.output + 'val.rec'
        fname_idx = args.output + 'val.idx'
        fname_lst = args.output + 'file_val.lst'

    record = mx.recordio.MXIndexedRecordIO(fname_idx, fname_rec, 'w')

    data_list = read_list(fname_lst)
    for idx, item in enumerate(data_list):
        data_encode(args, idx, item, q_out)

        if q_out.empty():
            continue
        _, s, _ = q_out.get()
        record.write_idx(item[0], s)


def read_worker(args, q_in, q_out):
    """Function that will be spawned to fetch the data
    from the input queue and put it back to output queue.
    Parameters
    ----------
    args: object
    q_in: queue
    q_out: queue
    """
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        data_encode(args, i, item, q_out)


def write_worker(q_out):
    """Function that will be spawned to fetch processed data
    from the output queue and write to the .rec file.
    Parameters
    ----------
    q_out: queue
    fname: string
    working_dir: string
    """
    count = 0

    if args.train:
        fname_rec = args.output + 'train.rec'
        fname_idx = args.output + 'train.idx'
    else:
        fname_rec = args.output + 'val.rec'
        fname_idx = args.output + 'val.idx'

    record = mx.recordio.MXIndexedRecordIO(fname_idx, fname_rec, 'w')

    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)
            count += 1


def create_recordIO_file_mp(args):
    logging.info("Creating a .rec and .idx file using multiprocessing...")
    if args.train:
        fname_lst = args.output + 'file_train.lst'
    else:
        fname_lst = args.output + 'file_val.lst'

    data_list = read_list(fname_lst)

    q_in = [multiprocessing.Queue(1024) for _ in range(args.num_thread)]
    q_out = multiprocessing.Queue(1024)

    # define the process
    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                    for i in range(args.num_thread)]

    # process data with num_thread process
    for p in read_process:
        p.start()

    # only use one process to write .rec to avoid race-condition
    write_process = multiprocessing.Process(target=write_worker, args=(q_out,))
    write_process.start()

    # put the data list into input queue
    for i, item in enumerate(data_list):
        q_in[i % len(q_in)].put((i, item))
    for q in q_in:
        q.put(None)
    for p in read_process:
        p.join()

    q_out.put(None)
    write_process.join()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='MXNet Benign and Malicious Example')

    parser.add_argument('--benign', type=str, required=True,
                        help='Benign training/validation folder')
    parser.add_argument('--malicious', type=str, required=True,
                        help='Malicious training/validation folder')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder to save .lst, .rec, and .idx file')
    parser.add_argument('--train', action='store_true', default=False,
                        help='This should be true if generating recordIO file for training')
    parser.add_argument('--val', action='store_true', default=False,
                        help='This should be true if generating recordIO file for validation')
    parser.add_argument('--feature-size', type=int, default=1024,
                        help="Feature encoding size of a data")
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='Shuffle the data')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be '
                             'different from the input list if >1. the input list will be '
                             'modified to match the resulting order.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    if not args.train and not args.val:
        raise ValueError(
            "Please pass either --train or --val command line argument. Run -h for help")

    create_list_file(args)

    if args.num_thread > 1 and multiprocessing is not None:
        create_recordIO_file_mp(args)
    else:
        create_recordIO_file(args)
