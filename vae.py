import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


# TODO: implement this if necessary
# def preprocess_mnist():


# parent class for all VAE variants
class AbstVAE:
    def __init__(self, model_scope, seed, checkpoint_dir, summary_dir, ):
        self.model_scope = model_scope
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir


    def build_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    # don't actually use this function, here for remembering the syntax
    def sample_noise(self, shape):
        # return np.random.randn(shape)
        return tf.random_normal(shape, seed=self.seed)


class VAE(AbstVAE):
    def __init__(self, model_scope, seed, checkpoint_dir, summary_dir, ):
        super().__init__(model_scope=model_scope, seed=seed, checkpoint_dir=checkpoint_dir, summary_dir=summary_dir)

    # what should dimensions of Gaussians be?
    def build_model(self):
        # TODO: data dims
        with tf.variable_scope(self.model_scope):
            # placeholders
            x = tf.placeholder(tf.float32, shape=[None], name="X")
            noise = tf.placeholder(tf.float32, shape=[None], name="noise")

            # set up network
            with tf.variable_scope("encoder"):
                # for now, hardcoding model architecture
                # TODO: allow for variable definition of model architecture

                # what activation function did they use?
                # constraining sigma to be a diagonal matrix?
                params = layers.fully_connected(x, num_outputs=500, activation_fn=tf.nn.sigmoid)
                params = layers.fully_connected(params, num_outputs=z_dim*2, activation_fn=tf.nn.sigmoid)
                mu = params[:z_dim]
                sigma = params[z_dim:]

            with tf.variable_scope("decoder"):
                # for now, hardcoding model architecture
                # TODO: allow for variable definition of model architecture

                z = mu + sigma * noise

                x_hat = layers.fully_connected(z, num_outputs=100, activation_fn=tf.nn.sigmoid)
                x_hat = layers.fully_connected(x_hat, num_outputs=x_dims, activation_fn=tf.nn.relu)  # ???

            reconstruction_loss = tf.nn.l2_loss(x_hat - x)
            kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)  # ???
            loss = reconstruction_loss + kl_loss


    def train(self):


# for debugging: in actual implementation, this should be separated
if __name__ == "__main__":
    train, test = tf.keras.datasets.mnist.load_data()
    mnist_x, mnist_y = train
