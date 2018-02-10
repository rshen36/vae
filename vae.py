import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
# import tensorflow.contrib.distributions as distributions


# parent class for all VAE variants
class AbstVAE:
    # def __init__(self, seed, experiment_dir, num_epochs, batch_size, model_scope):
    def __init__(self, seed, model_scope):
        self.seed = seed
        # self.experiment_dir = experiment_dir
        # self.num_epochs = num_epochs
        # self.batch_size = batch_size
        self.model_scope = model_scope
        np.random.seed(self.seed)  # set random seed elsewhere?

    # def encoder(self):

    # def decoder(self):

    # def build_graph(self):

    def _build_model(self):
        raise NotImplementedError

    # def sample(self, z):


class VAE(AbstVAE):
    def __init__(self, x_dims, z_dim=100, seed=123, model_name="vae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims
        self.z_dim = z_dim
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[None, int(np.prod(self.x_dims))], name="X")
        self.noise = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="noise")

        # set up network
        with tf.variable_scope("encoder"):
            # for now, hardcoding model architecture
            # TODO: allow for variable definition of model architecture

            # what activation function did they use?
            # constraining sigma to be a diagonal matrix?
            enet = layers.fully_connected(self.x, num_outputs=500, activation_fn=tf.nn.relu)
            enet = layers.fully_connected(enet, num_outputs=500, activation_fn=tf.nn.relu)
            params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None)
            mu = params[:, :self.z_dim]

            # TODO: taken from altosaar's implementation, change this
            sigma = 1e-6 + tf.nn.softplus(params[:, self.z_dim:])  # need to ensure std dev positive

        z = mu + sigma * self.noise

        with tf.variable_scope("decoder"):
            # for now, hardcoding model architecture
            # TODO: allow for variable definition of model architecture

            dnet = layers.fully_connected(z, num_outputs=500, activation_fn=tf.nn.relu)
            dnet = layers.fully_connected(dnet, num_outputs=500, activation_fn=tf.nn.relu)
            # ISSUE: x_hat appears to be saturating after some number of steps (not creating images anymore)
            # any point in making x_hat accessible? ability to sample images once model trained?
            self.x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)),
                                                activation_fn=tf.nn.sigmoid)  # ???

        reconstruction_loss = -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_hat) +
                                             (1 - self.x) * tf.log(1e-8 + 1 - self.x_hat), 1)  # ???
        kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)
        self.loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        xhat_img = tf.reshape(self.x_hat, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        tf.summary.image('reconstruction', xhat_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(reconstruction_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
