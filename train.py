import os
import argparse
import numpy as np
import tensorflow as tf

# from vae import VAE
from load_data import load_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vae', choices=['vae'],
                        help='type of variational autoencoder model (default: vae)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='dataset on which to train (default: mnist)')

    # TODO: input checks
    # parser.add_argument('--hparams_file', type=str, default='./hparams.json',
    #                     help='JSON file specifying the hyperparameters for training and record keeping')
    # parser.add_argument('--output_dir', type=str, default='./experiment',
    #                     help='directory to which to output training summary and checkpoint files')
    parser.add_argument('--seed', type=int, default=123, help='seed for rng')

    # TODO: add ability to pass hyperparameter values as a .json file
    # parser.add_argument('--num_epochs')
    # parser.add_argument('--batch_size')
    # parser.add_argument('--checkpoint_freq')
    # parser.add_argument('--lr')
    # parser.add_argument('--z_dim')

    return parser.parse_args()


def train():
    return


if __name__ == "__main__":
    args = parse_args()
    dataset = load_data()
    np.random.seed(args.seed)

    # output directories
    # anything else that should be outputted/recorded? logs? example images?
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    summary_dir = os.path.join(args.experiment_dir, "summaries")
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # ISSUE: how best to allow for variable specification of the model?
    # TODO: taken from hwalsuklee's tensorflow generative model collection project, change this
    # models = [VAE]
    models = []

    # TODO: look into session config options
    with tf.Session() as sess:
        for model in models:
            if args.model == model:
                model = model()

        global_step = 0
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)  # TODO: fix this

        num_steps = dataset.train.num_examples // args.batch_size
        # for epoch in range(args.num_epochs):
        #     for step in range(num_steps):
        while dataset.train.epochs_completed < args.num_epochs:
            # Dataset class keeps track of steps in current epoch and number epochs elapsed
            batch = dataset.train.next_batch(args.batch_size)
            summary, _ = sess.run(
                [model.merged, model.train_op],  # TODO: fix this
                feed_dict={
                    # TODO: fix this
                    model.x: batch[0],
                    model.noise: np.random.randn(args.batch_size, args.z_dim)  # TODO: set global random seed
                })
            global_step += 1

            summary_writer.add_summary(summary, global_step)
            summary_writer.flush()

            if dataset.train.epochs_completed % args.checkpoint_freq == 0:
                saver.save(sess, checkpoint_path, global_step=global_step)