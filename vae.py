import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as dbns


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


class BernoulliVAE(AbstVAE):
    def __init__(self, x_dims, z_dim=100, hidden_dim=500, lr=.02, batch_size=100, seed=123, model_name="vae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims  # TODO: figure out how to deal with channels/color images
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, int(np.prod(self.x_dims))], name="X")
        self.p_z = dbns.Normal(loc=tf.zeros(shape=[self.batch_size, self.z_dim], dtype=tf.float32),
                               scale=tf.zeros(shape=[self.batch_size, self.z_dim], dtype=tf.float32))

        # set up network
        with tf.variable_scope("encoder"):
            # for now, hardcoding model architecture as that specified in paper
            # TODO: allow for variable definition of model architecture

            enet = layers.fully_connected(self.x, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z_params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z_mu = z_params[:, self.z_dim:]
            z_sigma = tf.exp(z_params[:, :self.z_dim])
            self.q_z = dbns.Normal(loc=z_mu, scale=z_sigma)

        z = z_mu + tf.multiply(z_sigma, self.p_z.sample())
        # z = self.q_z.sample()

        with tf.variable_scope("decoder"):
            # for now, hardcoding model architecture as that specified in paper
            # TODO: allow for variable definition of model architecture

            dnet = layers.fully_connected(z, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)),
                                                activation_fn=tf.nn.sigmoid,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                                )  # Bernoulli MLP decoder
            self.p_x_z = dbns.Bernoulli(logits=self.x_hat)

        nll_loss = -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_hat) +
                                  (1 - self.x) * tf.log(1e-8 + 1 - self.x_hat), 1)  # Bernoulli nll
        kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(1e-8 + tf.square(z_sigma)) - 1, 1)
        # kl_loss = tf.reduce_sum(dbns.kl_divergence(self.q_z, self.p_z), 1)
        self.loss = tf.reduce_mean(nll_loss + kl_loss)
        self.elbo = -1.0 * tf.reduce_mean(nll_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        xhat_img = tf.reshape(self.x_hat, [-1] + self.x_dims)
        tf.summary.image('reconstruction', xhat_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(nll_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()


class GaussianVAE(AbstVAE):
    def __init__(self, x_dims, z_dim=100, hidden_dim=500, lr=.02, batch_size=100, seed=123, model_name="vae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims  # TODO: figure out how to deal with channels/color images
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, int(np.prod(self.x_dims))], name="X")
        self.p_z = dbns.Normal(loc=tf.zeros(shape=[self.batch_size, self.z_dim], dtype=tf.float32),
                               scale=tf.ones(shape=[self.batch_size, self.z_dim], dtype=tf.float32))

        # set up network
        with tf.variable_scope("encoder"):
            # for now, hardcoding model architecture as that specified in paper
            # TODO: allow for variable definition of model architecture
            enet = layers.fully_connected(self.x, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z_params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            z_mu = z_params[:, self.z_dim:]
            z_sigma = tf.exp(z_params[:, :self.z_dim])
            self.q_z = dbns.Normal(loc=z_mu, scale=z_sigma)

        z = z_mu + tf.multiply(z_sigma, self.p_z.sample())
        # z = self.q_z.sample()

        with tf.variable_scope("decoder"):
            # for now, hardcoding model architecture as that specified in paper
            # TODO: allow for variable definition of model architecture
            dnet = layers.fully_connected(z, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            out_params = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)*2), activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            mu = tf.nn.sigmoid(out_params[:, int(np.prod(self.x_dims)):])  # out_mu constrained to (0,1)
            sigma = tf.exp(out_params[:, :int(np.prod(self.x_dims))])
            self.p_x_z = dbns.Normal(loc=mu, scale=tf.sqrt(sigma))

        nll_loss = (tf.log(2.0 * np.pi) * (0.5 * self.batch_size)) + \
            (0.5 * tf.reduce_sum(tf.log(1e-8 + tf.square(sigma)), 1)) + \
            tf.reduce_sum(tf.square(self.x - mu) / (2.0 * tf.square(sigma)), 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(1e-8 + tf.square(z_sigma)) - 1, 1)
        # nll_loss = -tf.reduce_sum(self.p_x_z.log_prob(self.x), 1)
        # kl_loss = tf.reduce_sum(dbns.kl_divergence(self.q_z, self.p_z), 1)

        self.loss = tf.reduce_mean(nll_loss + kl_loss)
        self.elbo = -1.0 * tf.reduce_mean(nll_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        sample_img = tf.reshape(mu, [-1] + self.x_dims)
        tf.summary.image('samples', sample_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(nll_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()


class BernoulliIWAE(AbstVAE):
    def __init__(self, x_dims, z_dim=50, hidden_dim=200, seed=123, batch_size=20, n_samples=5, model_name="iwae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims  # TODO: figure out how to deal with channels/color images
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_samples = n_samples
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, int(np.prod(self.x_dims))], name="X")
        x = tf.tile(self.x, multiples=[self.n_samples, 1])
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")

        # okay to sample this way?
        self.p_z = dbns.Normal(loc=tf.zeros(shape=[self.batch_size * self.n_samples, self.z_dim]),
                               scale=tf.ones(shape=[self.batch_size * self.n_samples, self.z_dim]))

        # set up network
        with tf.variable_scope("encoder"):
            # Initialization via heuristic specified by Glorot & Bengio 2010
            # should the seed be set with the initializers?
            enet = layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer())
            enet = layers.fully_connected(enet, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer())
            z_params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None,
                                              weights_initializer=layers.xavier_initializer(),
                                              biases_initializer=layers.xavier_initializer())
            z_mu = z_params[:, self.z_dim:]
            z_sigma = tf.exp(z_params[:, :self.z_dim])
            self.q_z = dbns.Normal(loc=z_mu, scale=z_sigma)  # did they predict var or stddev?

        z = z_mu + tf.multiply(z_sigma, self.p_z.sample())

        with tf.variable_scope("decoder"):
            # Initialization via heuristic specified by Glorot & Bengio 2010
            # should the seed be set with the initializers?
            dnet = layers.fully_connected(z, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer())
            dnet = layers.fully_connected(dnet, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer())
            x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)), activation_fn=tf.nn.sigmoid,
                                           weights_initializer=layers.xavier_initializer(),
                                           biases_initializer=layers.xavier_initializer()
                                           )  # Bernoulli MLP decoder
            self.out_dbn = dbns.Bernoulli(logits=x_hat)

        log_lik = tf.reduce_sum(x * tf.log(1e-8 + x_hat) + (1 - x) * tf.log(1e-8 + 1 - x_hat), 1)
        neg_kld = tf.reduce_sum(self.p_z.log_prob(z) - self.q_z.log_prob(z), 1)

        # calculate importance weights using logsumexp and exp-normalize tricks
        log_iws = tf.reshape(log_lik, [self.batch_size, self.n_samples]) + \
            tf.reshape(neg_kld, [self.batch_size, self.n_samples])
        max_log_iws = tf.reduce_max(log_iws, axis=1, keepdims=True)
        self.elbo = tf.reduce_mean(max_log_iws + tf.log(tf.reduce_mean(
            tf.exp(log_iws - max_log_iws), axis=1, keepdims=True)))
        self.loss = -self.elbo

        # for now, hardcoding the Adam optimizer parameters used in the paper
        # necessary to modify gradients for importance weighting? reparameterization trick takes care of it?
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=0.0001)
        self.train_op = optimizer.minimize(self.loss)

        # tensorboard summaries
        x_img = tf.reshape(x, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        sample_img = tf.reshape(x_hat, [-1] + self.x_dims)
        tf.summary.image('samples', sample_img)
        tf.summary.scalar('log_lik', tf.reduce_mean(log_lik))
        tf.summary.scalar('neg_kld', tf.reduce_mean(neg_kld))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()


