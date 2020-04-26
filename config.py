import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataname', 'mirflickr','dataset name')
tf.app.flags.DEFINE_string('model_dir', './model/','path to store the checkpoints of the model')
tf.app.flags.DEFINE_string('summary_dir', './summary','path to store analysis summaries used for tensorboard')
tf.app.flags.DEFINE_string('checkpoint_path', './model/model-43248','The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string('data_dir', '../data/mirflickr/mirflickr_data.npy','The path of input observation data')
tf.app.flags.DEFINE_string('train_idx', '../data/mirflickr/mirflickr_train_idx.npy','The path of training data index')
tf.app.flags.DEFINE_string('valid_idx', '../data/mirflickr/mirflickr_val_idx.npy','The path of validation data index')
tf.app.flags.DEFINE_string('test_idx', '../data/mirflickr/mirflickr_test_idx.npy','The path of testing data index')
tf.app.flags.DEFINE_string('test_sh_path', './run_test_ebird.sh', 'run test bash')
tf.app.flags.DEFINE_string('saved_ckpt', '', 'restore saved checkpoint')


tf.app.flags.DEFINE_integer('batch_size', 128, 'the number of data points in one minibatch')
tf.app.flags.DEFINE_integer('testing_size', 128, 'the number of data points in one testing or validation batch') 
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate')
tf.app.flags.DEFINE_integer('max_epoch', 200, 'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay', 0.00001, 'weight decay rate')
tf.app.flags.DEFINE_float('lr_decay_ratio', 0.5, 'The decay ratio of learning rate')
tf.app.flags.DEFINE_float('lr_decay_times', 3.0, 'How many times does learning rate decay')
tf.app.flags.DEFINE_integer('n_test_sample', 10000, 'The sampling times for the testing')
tf.app.flags.DEFINE_integer('n_train_sample', 100, 'The sampling times for the training') 

tf.app.flags.DEFINE_integer('z_dim', 100, 'z dimention: the number of the independent normal random variables in DMSE \
    / the rank of the residual covariance matrix')

tf.app.flags.DEFINE_integer('label_dim', 100, 'the number of labels in current training')
tf.app.flags.DEFINE_integer('latent_dim', 50, 'the number of labels in current training') 
tf.app.flags.DEFINE_integer('meta_offset', 0, 'the offset caused by meta data') 
tf.app.flags.DEFINE_integer('feat_dim', 15, 'the dimensionality of the features ')

tf.app.flags.DEFINE_float('save_epoch', 1.0, 'epochs to save the checkpoint of the model')
tf.app.flags.DEFINE_integer('max_keep', 3, 'maximum number of saved model')
tf.app.flags.DEFINE_integer('check_freq', 120, 'checking frequency')

tf.app.flags.DEFINE_float('nll_coeff', 0.1, "nll_loss coefficient")
tf.app.flags.DEFINE_float('l2_coeff', 1.0, "l2_loss coefficient")
tf.app.flags.DEFINE_float('c_coeff', 200., "c_loss coefficient")
tf.app.flags.DEFINE_float('scale_coeff', 1.0, "mu/logvar scale coefficient")
tf.app.flags.DEFINE_float('keep_prob', 0.5, "drop out rate")
tf.app.flags.DEFINE_boolean('resume', False, "whether to resume a ckpt")
tf.app.flags.DEFINE_boolean('write_to_test_sh', False, 'whether to modify test.sh')

