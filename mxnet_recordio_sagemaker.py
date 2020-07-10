import os
import re
import time
import logging
import argparse
import json
import pickle

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, recordio


def train(args):
    class CustomRecordDataset(gluon.data.dataset.RecordFileDataset):
        """A custom dataset wrapping over a RecordIO file containing html data.

        Each sample is a feature representation of html data and its corresponding label.

        Parameters
        ----------
        filename : str
            Path to rec file.

        """

        def __init__(self, filename):
            super(CustomRecordDataset, self).__init__(filename)

        def __getitem__(self, idx):
            record = super(CustomRecordDataset, self).__getitem__(idx)
            header, img = recordio.unpack(record)
            img = pickle.loads(img)
            return mx.nd.array(img), np.array(header.label, dtype=np.float32)

    # Function to get train and val dataloader
    def get_dataloader():
        train_datapath = os.path.join(args.train, 'output-train.rec')
        train_dataset = CustomRecordDataset(train_datapath)
        train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, shuffle=True)

        val_datapath = os.path.join(args.train, 'output-val.rec')
        val_dataset = CustomRecordDataset(val_datapath)
        val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False)

        return train_dataloader, val_dataloader

    # Function to define neural network
    def custom_model():
        net = gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(1024, activation='relu'))
            net.add(gluon.nn.Dense(512, activation='relu'))
            net.add(gluon.nn.Dense(1, activation='sigmoid'))
        return net

    # Function to get binary labels
    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    # Function to evaluate accuracy for a model
    def evaluate(model, val_data, ctx):
        metric = mx.metric.CustomMetric(facc)
        for data, label in val_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = model(data)
            metric.update(label, output)

        return metric.get()

    # Function to train the model
    def train_model(net, train_dataloader, val_dataloader):
        train_metric = mx.metric.CustomMetric(facc)
        for epoch in range(args.epochs):
            tic = time.time()
            # reset metric at beginning of epoch.
            train_metric.reset()
            for i, (data, label) in enumerate(train_dataloader):
                # Copy data to ctx if necessary
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)

                # Start recording computation graph with record() section.
                # Recorded graphs can then be differentiated with backward.
                with autograd.record():
                    output = net(data)
                    L = loss_fn(output, label)
                L.backward()
                curr_loss = mx.ndarray.mean(L).asscalar()

                # take a gradient step with batch_size equal to data.shape[0]
                trainer.step(args.batch_size)
                # update metric at last.
                train_metric.update(label, output)

                if i % args.log_interval == 0:
                    name, acc = train_metric.get()
                    logging.info('[Epoch %d Batch %d] Training_Loss: %f Training_Acc: %f' %
                                 (epoch, i, curr_loss, acc))
            elapsed = time.time() - tic
            speed = i * args.batch_size / elapsed
            logging.info('Epoch[%d]\tSpeed=%.2f samples/sec \tTime cost=%f secs',
                         epoch, speed, elapsed)

            # Evaluate the model
            if not (epoch + 1) % args.val_interval:
                val_name, val_acc = evaluate(net, val_dataloader, ctx)
                logging.info('Validation Accuracy: %f' % (val_acc))

    # Fixed the seed for randomness
    mx.random.seed(args.seed)

    # Get the context
    if args.cuda and mx.context.num_gpus() > 0:
        logging.info("Running the script on single GPU")
        ctx = mx.gpu(0)
    else:
        logging.info("Running the script on CPU")
        ctx = mx.cpu()

    # Create a model
    net = custom_model()
    net.cast(args.dtype)
    net.hybridize(static_alloc=True, static_shape=True)

    # Initialize parameters
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                 magnitude=2)
    net.initialize(initializer, ctx=ctx)

    # Create optimizer
    optimizer_params = {'learning_rate': args.lr, 'momentum': args.momentum}

    opt = mx.optimizer.create('sgd', **optimizer_params)
    trainer = gluon.Trainer(net.collect_params(), opt)
    loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    train_dataloader, val_dataloader = get_dataloader()

    train_model(net, train_dataloader, val_dataloader)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='MXNet Benign and Malicious Example')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size (default: 128)')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs '
                                        'are powerful.')
    parser.add_argument('--rec-train', type=str, default='',
                        help='the training data in recordio format')
    parser.add_argument('--rec-val', type=str, default='',
                        help='the validation data in recordio format')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='training data type (default: float32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate (default: 0.02)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Train on GPU with CUDA')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed to be fixed.')

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    num_gpus = int(os.environ['SM_NUM_GPUS'])

    train(args)
