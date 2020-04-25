import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
FLAGS = tf.app.flags.FLAGS

class MODEL:

    def build_multi_classify_loss(self, predictions, labels):
        shape = tf.shape(labels)

        labels = tf.cast(labels, tf.float32) # labels: n_batch * n_labels, e.g. 128*100
        y_i = tf.equal(labels, tf.ones(shape)) # turn ones in labels to True, 128*100
        y_not_i = tf.equal(labels, tf.zeros(shape)) # turn zeros in labels to True, 128*100

        # get indices to check
        truth_matrix = tf.compat.v1.to_float(self.pairwise_and(y_i, y_not_i)) # pairs of 0/1 of labels for one sample, 128*100*100

        # calculate all exp'd differences
        # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
        sub_matrix = self.pairwise_sub(predictions, predictions) # pairwise subtraction, 100*128*100*100
        exp_matrix = tf.exp(tf.negative(5 * sub_matrix)) # take the exponential, 100*128*100*100

        # check which differences to consider and sum them
        sparse_matrix = tf.multiply(exp_matrix, truth_matrix) # zero-out the ones with the same label, 100*128*100*100
        sums = tf.reduce_sum(sparse_matrix, axis=[2, 3]) # loss for each sample in every batch, 100*128

        # get normalizing terms and apply them
        y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1) # number of 1's for each sample, 128
        y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_not_i), axis=1) # number of 0's, 128
        normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes) # 128

        loss = tf.divide(sums, 5*normalizers) # 100*128  divide  128
        zero = tf.zeros_like(loss) # 100*128 zeros
        loss = tf.where(tf.logical_or(tf.math.is_inf(loss), tf.math.is_nan(loss)), x=zero, y=loss)
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_mean(loss)

        return loss

    def pairwise_and(self, a, b):
        """compute pairwise logical and between elements of the tensors a and b
        Description
        -----
        if y shape is [3,3], y_i would be translate to [3,3,1], y_not_i is would be [3,1,3]
        and return [3,3,3],through the matrix ,we can easy to caculate c_k - c_i(appear in the paper)
        """
        column = tf.expand_dims(a, 2)
        row = tf.expand_dims(b, 1)
        return tf.logical_and(column, row)
    
    def pairwise_sub(self, a, b):
        """compute pairwise differences between elements of the tensors a and b
        :param a:
        :param b:
        :return:
        """
        column = tf.expand_dims(a, 3)
        row = tf.expand_dims(b, 2)
        return tf.subtract(column, row)

    def cross_entropy_loss(self, logits, labels, n_sample):
        labels = tf.tile(tf.expand_dims(labels, 0), [n_sample, 1, 1])
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        ce_loss = tf.reduce_mean(tf.reduce_sum(ce_loss, axis=1))
        return ce_loss

    def __init__(self, is_training, cholesky=None):

        tf.compat.v1.set_random_seed(19940423)

        label_dim = FLAGS.label_dim # number of labels
        latent_dim = FLAGS.latent_dim # the dimension of the latent space
        
        self.input_feat = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,FLAGS.feat_dim],name='input_feat')
        self.input_label = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,FLAGS.label_dim],name='input_label')

        self.keep_prob = tf.compat.v1.placeholder(tf.float32) #keep probability for the dropout
        weights_regularizer = slim.l2_regularizer(FLAGS.weight_decay)


        ## label encoder
        # we concatenate features with labels in this implementation, since this made the training more stable. similar techniques used in Conditional VAE
        input_x = tf.concat([self.input_feat, self.input_label], 1)
        self.fe_1 = slim.dropout(slim.fully_connected(input_x, 512, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='label_encoder/fc_1'), keep_prob=self.keep_prob, is_training=is_training)
        self.fe_2 = slim.dropout(slim.fully_connected(self.fe_1, 256, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='label_encoder/fc_2'), keep_prob=self.keep_prob, is_training=is_training)
        self.fe_mu = slim.fully_connected(self.fe_2, FLAGS.latent_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_miu') * FLAGS.scale_coeff
        self.fe_logvar = slim.fully_connected(self.fe_2, FLAGS.latent_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_logvar') * FLAGS.scale_coeff
        eps = tf.random.normal(shape=tf.shape(self.fe_mu))
        fe_sample = eps * tf.exp(self.fe_logvar / 2) + self.fe_mu

        ## feature encoder (informative prior)
        self.fx_1 = slim.dropout(slim.fully_connected(self.input_feat, 256, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='feat_encoder/fc_1'), keep_prob=self.keep_prob, is_training=is_training)
        self.fx_2 = slim.dropout(slim.fully_connected(self.fx_1, 512, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='feat_encoder/fc_2'), keep_prob=self.keep_prob, is_training=is_training)
        self.fx_3 = slim.dropout(slim.fully_connected(self.fx_2, 256, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='feat_encoder/fc_3'), keep_prob=self.keep_prob, is_training=is_training)
        self.fx_mu = slim.fully_connected(self.fx_3, FLAGS.latent_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='feat_encoder/z_miu') * FLAGS.scale_coeff
        self.fx_logvar = slim.fully_connected(self.fx_3, FLAGS.latent_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='feat_encoder/z_logvar') * FLAGS.scale_coeff
        fx_sample = eps * tf.exp(self.fx_logvar / 2) + self.fx_mu

        # kl divergence between two learnt normal distributions
        self.kl_loss = tf.reduce_mean(0.5*tf.reduce_sum((self.fx_logvar-self.fe_logvar)-1+tf.exp(self.fe_logvar-self.fx_logvar)+tf.divide(tf.pow(self.fx_mu-self.fe_mu, 2), tf.exp(self.fx_logvar)+1e-6), axis=1))

        # concatenate input_feat with samples. similar technique in Conditional VAE
        c_fe_sample = tf.concat([self.input_feat, fe_sample], 1)
        c_fx_sample = tf.concat([self.input_feat, fx_sample], 1)
        

        ## label decoder
        self.fd_1 = slim.fully_connected(c_fe_sample, 256, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='label_decoder/fc_1')
        self.fd_2 = slim.fully_connected(self.fd_1, 512, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, scope='label_decoder/fc_2')

        ## feature decoder
        self.fd_x_1 = slim.fully_connected(c_fx_sample, 256, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, reuse=True, scope='label_decoder/fc_1')
        self.fd_x_2 = slim.fully_connected(self.fd_x_1, 512, weights_regularizer=weights_regularizer, activation_fn=tf.nn.relu, reuse=True, scope='label_decoder/fc_2')

        
        # derive the label mean in the Multivariate Probit model
        self.label_mp_mu = slim.fully_connected(self.fd_2, label_dim, activation_fn=None, weights_regularizer=weights_regularizer, scope='label_mp_mu')

        # derive the feature mean in the Multivariate Probit model
        self.feat_mp_mu = slim.fully_connected(self.fd_x_2, label_dim, activation_fn=None, weights_regularizer=weights_regularizer, scope='feat_mp_mu')


        # initialize the square root of the residual covariance matrix 
        self.r_sqrt_sigma=tf.Variable(np.random.uniform(-np.sqrt(6.0/(label_dim+FLAGS.z_dim)), np.sqrt(6.0/(label_dim+FLAGS.z_dim)), (label_dim, FLAGS.z_dim)), dtype=tf.float32, name='r_sqrt_sigma')
        # construct a semi-positive definite matrix
        self.sigma=tf.matmul(self.r_sqrt_sigma, tf.transpose(self.r_sqrt_sigma))

        # covariance = residual_covariance + identity
        self.covariance=self.sigma + tf.eye(label_dim)
        
        # epsilon
        self.eps1=tf.constant(1e-6, dtype="float32")

        n_sample = FLAGS.n_train_sample
        if (is_training==False):
            n_sample = FLAGS.n_test_sample

        # batch_size
        n_batch = tf.shape(self.label_mp_mu)[0]

        # standard Gaussian samples
        self.noise = tf.random.normal(shape=[n_sample, n_batch, FLAGS.z_dim])
        
        # see equation (3) in the paper for this block
        self.B = tf.transpose(self.r_sqrt_sigma)
        self.sample_r = tf.tensordot(self.noise, self.B, axes=1)+self.label_mp_mu #tensor: n_sample*n_batch*label_dim
        self.sample_r_x = tf.tensordot(self.noise, self.B, axes=1)+self.feat_mp_mu #tensor: n_sample*n_batch*label_dim
        norm=tf.distributions.Normal(0., 1.)
        
        # the probabilities w.r.t. every label in each sample from the batch
        # size: n_sample * n_batch * label_dim
        # eps1: to ensure the probability is non-zero
        E = norm.cdf(self.sample_r)*(1-self.eps1)+self.eps1*0.5
        # similar for the feature branch
        E_x = norm.cdf(self.sample_r_x)*(1-self.eps1)+self.eps1*0.5

        def compute_BCE_and_RL_loss(E):
            #compute negative log likelihood (BCE loss) for each sample point
            sample_nll = tf.negative((tf.math.log(E)*self.input_label+tf.math.log(1-E)*(1-self.input_label)), name='sample_nll')
            logprob=-tf.reduce_sum(sample_nll, axis=2)

            #the following computation is designed to avoid the float overflow (log_sum_exp trick)
            maxlogprob=tf.reduce_max(logprob, axis=0)
            Eprob=tf.reduce_mean(tf.exp(logprob-maxlogprob), axis=0)
            nll_loss=tf.reduce_mean(-tf.math.log(Eprob)-maxlogprob)

            # compute the ranking loss (RL loss) 
            c_loss = self.build_multi_classify_loss(E, self.input_label)
            return nll_loss, c_loss

        # BCE and RL losses for label branch
        self.nll_loss, self.c_loss = compute_BCE_and_RL_loss(E)
        # BCE and RL losses for feature branch
        self.nll_loss_x, self.c_loss_x = compute_BCE_and_RL_loss(E_x)
           
        # if in the training phase, the prediction 
        self.indiv_prob = tf.reduce_mean(E_x, axis=0, name='individual_prob')

        # weight regularization
        self.l2_loss = tf.add_n(tf.compat.v1.losses.get_regularization_losses())

        # total loss: refer to equation (5)
        self.total_loss = self.l2_loss * FLAGS.l2_coeff + (self.nll_loss + self.nll_loss_x) * FLAGS.nll_coeff + (self.c_loss + self.c_loss_x) * FLAGS.c_coeff + self.kl_loss * 1.1
        
