import tensorflow as tf
from .rbm import RBM
from .util import sample_bernoulli, sample_gaussian


class GBRBM(RBM):
    def __init__(self, n_visible, n_hidden, sample_visible=False, sigma=1, **kwargs):
        self.sample_visible = sample_visible
        self.sigma = sigma
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        RBM.__init__(self, n_visible, n_hidden, **kwargs)

    def _initialize_vars(self):
        #self.hidden_p = tf.nn.relu(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.hidden_p=tf.nn.relu(tf.matmul(self.x,self.w) + self.hidden_bias + tf.random_normal(seed=1,shape=[1, self.n_hidden],mean=tf.zeros([1, self.n_hidden],tf.float32),stddev=tf.sigmoid(tf.matmul(self.x,self.w) + self.hidden_bias)))
        #self.visible_recon_p = tf.matmul(sample_bernoulli(self.hidden_p), tf.transpose(self.w)) + self.delta_visible_bias
        self.visible_recon_p = tf.random_normal(seed=1,shape=[1, self.n_visible],mean=tf.matmul(self.hidden_p,tf.transpose(self.w))+self.visible_bias,stddev=1)
        if self.sample_visible:
            self.visible_recon_p = sample_gaussian(self.visible_recon_p, self.sigma)  
        self.hidden_recon_p = tf.nn.relu(tf.matmul(self.visible_recon_p,self.w) + self.hidden_bias + tf.random_normal(seed=1,shape=[1, self.n_hidden],mean=tf.zeros([1, self.n_hidden],tf.float32),stddev=tf.sigmoid(tf.matmul(self.visible_recon_p,self.w) + self.hidden_bias)))
        #self.hidden_recon_p = tf.nn.relu(tf.matmul(self.visible_recon_p, self.w) + self.hidden_bias)
        positive_grad = tf.matmul(tf.transpose(self.x), self.hidden_p)
        negative_grad = tf.matmul(tf.transpose(self.visible_recon_p), self.hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - self.visible_recon_p, 0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(self.hidden_p - self.hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.relu(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias
        self.compute_visible_from_hidden = tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias
