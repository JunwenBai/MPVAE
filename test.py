import tensorflow as tf
import numpy as np
import model
import get_data 
import config 
import sys
import evals
sys.path.append("./")

FLAGS = tf.app.flags.FLAGS

THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]
METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']

def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('reading npy...')

    data = np.load(FLAGS.data_dir)
    test_idx = np.load(FLAGS.test_idx)

    print('reading completed')

    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=session_config)

    print('building network...')

    classifier = model.MODEL(is_training=False)
    global_step = tf.Variable(0,name='global_step',trainable=False)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, FLAGS.checkpoint_path)
    model_id = FLAGS.checkpoint_path.split("-")[-1]

    print('restoring from '+FLAGS.checkpoint_path)

    def test_step(test_idx, name="Test"):
        print('{}...'.format(name))
        all_nll_loss = 0
        all_l2_loss = 0
        all_c_loss = 0
        all_total_loss = 0

        all_indiv_prob = []
        all_label = []
        all_indiv_max = []

        sigma=[]
        real_batch_size=min(FLAGS.testing_size, len(test_idx))
        
        N_test_batch = int( (len(test_idx)-1)/real_batch_size ) + 1

        for i in range(N_test_batch):
            if i % 20 == 0:
                print("%.1f%% completed" % (i*100.0/N_test_batch))

            start = real_batch_size*i
            end = min(real_batch_size*(i+1), len(test_idx))

            input_feat = get_data.get_feat(data,test_idx[start:end])
            input_label = get_data.get_label(data,test_idx[start:end])

            feed_dict={}
            feed_dict[classifier.input_feat]=input_feat
            feed_dict[classifier.input_label]=input_label
            feed_dict[classifier.keep_prob]=1.0

            nll_loss, l2_loss, c_loss, total_loss, indiv_prob, covariance = sess.run([classifier.nll_loss, classifier.l2_loss, classifier.c_loss, \
                classifier.total_loss, classifier.indiv_prob, classifier.covariance], feed_dict)

            all_nll_loss += nll_loss*(end-start)
            all_l2_loss += l2_loss*(end-start)
            all_c_loss += c_loss*(end-start)
            all_total_loss += total_loss*(end-start)

            if (all_indiv_prob == []):
                all_indiv_prob = indiv_prob
            else:
                all_indiv_prob = np.concatenate((all_indiv_prob, indiv_prob))

            if (all_label == []):
                all_label = input_label
            else:
                all_label = np.concatenate((all_label, input_label))

        nll_loss = all_nll_loss / len(test_idx)
        l2_loss = all_l2_loss / len(test_idx)
        c_loss = all_c_loss / len(test_idx)
        total_loss = all_total_loss / len(test_idx)
        return all_indiv_prob, all_label

    indiv_prob, input_label = test_step(test_idx, "Test")
    n_label = indiv_prob.shape[1]

    best_test_metrics = None
    for threshold in THRESHOLDS:
        test_metrics = evals.compute_metrics(indiv_prob, input_label, threshold, all_metrics=True)
        if best_test_metrics == None:
            best_test_metrics = {}
            for metric in METRICS:
                best_test_metrics[metric] = test_metrics[metric]
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_test_metrics[metric] = min(best_test_metrics[metric], test_metrics[metric])
                else:
                    best_test_metrics[metric] = max(best_test_metrics[metric], test_metrics[metric])

    print("****************")
    for metric in METRICS:
        print(metric, ":", best_test_metrics[metric])
    print("****************")


if __name__=='__main__':
    tf.app.run()

