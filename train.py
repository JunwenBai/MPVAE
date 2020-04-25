import tensorflow as tf
import numpy as np
import datetime
import model
import get_data
import config 
from utils import build_path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import sys
import evals
sys.path.append('./')

FLAGS = tf.app.flags.FLAGS

THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]

METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']

def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.compat.v1.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def train_step(sess, hg, merged_summary, summary_writer, input_label, input_feat, train_op, global_step):
    feed_dict={}
    feed_dict[hg.input_feat]=input_feat
    feed_dict[hg.input_label]=input_label
    feed_dict[hg.keep_prob]=FLAGS.keep_prob

    temp, step, c_loss, c_loss_x, nll_loss, nll_loss_x, l2_loss, kl_loss, total_loss, summary, indiv_prob = \
    sess.run([train_op, global_step, hg.c_loss, hg.c_loss_x, hg.nll_loss, hg.nll_loss_x, hg.l2_loss, hg.kl_loss, hg.total_loss, merged_summary, hg.indiv_prob], feed_dict)

    train_metrics = evals.compute_metrics(indiv_prob, input_label, 0.5, all_metrics=False)
    macro_f1, micro_f1 = train_metrics['maF1'], train_metrics['miF1']

    summary_writer.add_summary(MakeSummary('train/nll_loss', nll_loss),step)
    summary_writer.add_summary(MakeSummary('train/l2_loss', l2_loss),step)
    summary_writer.add_summary(MakeSummary('train/c_loss', c_loss),step)
    summary_writer.add_summary(MakeSummary('train/total_loss', total_loss),step)
    summary_writer.add_summary(MakeSummary('train/macro_f1', macro_f1),step)
    summary_writer.add_summary(MakeSummary('train/micro_f1', micro_f1),step)

    return indiv_prob, nll_loss, nll_loss_x, l2_loss, c_loss, c_loss_x, kl_loss, total_loss, macro_f1, micro_f1


def validation_step(sess, hg, data, merged_summary, summary_writer, valid_idx, global_step, mode='val'):
    print('%s...'%mode)

    all_nll_loss = 0
    all_l2_loss = 0
    all_c_loss = 0
    all_total_loss = 0

    all_indiv_prob = []
    all_label = []

    real_batch_size=min(FLAGS.batch_size, len(valid_idx))
    for i in range(int( (len(valid_idx)-1)/real_batch_size )+1):
        start = real_batch_size*i
        end = min(real_batch_size*(i+1), len(valid_idx))

        input_feat = get_data.get_feat(data,valid_idx[start:end])
        input_label = get_data.get_label(data,valid_idx[start:end])

        feed_dict={}
        feed_dict[hg.input_feat]=input_feat
        feed_dict[hg.input_label]=input_label
        feed_dict[hg.keep_prob]=1.0

        nll_loss, l2_loss, c_loss, total_loss, indiv_prob = sess.run([hg.nll_loss, hg.l2_loss, hg.c_loss, hg.total_loss, hg.indiv_prob], feed_dict)
    
        all_nll_loss += nll_loss*(end-start)
        all_l2_loss += l2_loss*(end-start)
        all_c_loss += c_loss*(end-start)
        all_total_loss += total_loss*(end-start)
    
        for i in indiv_prob:
            all_indiv_prob.append(i)
        for i in input_label:
            all_label.append(i)

    # collect all predictions and ground-truths
    all_indiv_prob = np.array(all_indiv_prob)
    all_label = np.array(all_label)

    nll_loss = all_nll_loss/len(valid_idx)
    l2_loss = all_l2_loss/len(valid_idx)
    c_loss = all_c_loss/len(valid_idx)
    total_loss = all_total_loss/len(valid_idx)

    best_val_metrics = None
    if mode == 'val':
        for threshold in THRESHOLDS:
            val_metrics = evals.compute_metrics(all_indiv_prob, all_label, threshold, all_metrics=True)

            if best_val_metrics == None:
                best_val_metrics = {}
                for metric in METRICS:
                    best_val_metrics[metric] = val_metrics[metric]
            else:
                for metric in METRICS:
                    if 'FDR' in metric:
                        best_val_metrics[metric] = min(best_val_metrics[metric], val_metrics[metric])
                    else:
                        best_val_metrics[metric] = max(best_val_metrics[metric], val_metrics[metric])

    time_str = datetime.datetime.now().isoformat()
    acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], best_val_metrics['ebF1'], best_val_metrics['maF1'], best_val_metrics['miF1']

    # nll_coeff: BCE coeff, lambda_1
    # c_coeff: Ranking loss coeff, lambda_2
    # l2_coeff: weight decay
    print("**********************************************")
    print("%s results: %s\nacc=%.6f\tha=%.6f\texam_f1=%.6f, macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tl2_loss=%.6f\tc_loss=%.6f\ttotal_loss=%.6f" % (mode, time_str, acc, ha, ebf1, maf1, mif1, nll_loss*FLAGS.nll_coeff, l2_loss*FLAGS.l2_coeff, c_loss*FLAGS.c_coeff, total_loss))
    print("**********************************************")


    current_step = sess.run(global_step) #get the value of global_step
    summary_writer.add_summary(MakeSummary('%s/nll_loss' % mode, nll_loss), current_step)
    summary_writer.add_summary(MakeSummary('%s/l2_loss' % mode, l2_loss), current_step)
    summary_writer.add_summary(MakeSummary('%s/c_loss' % mode, c_loss), current_step)
    summary_writer.add_summary(MakeSummary('%s/total_loss' % mode,total_loss), current_step)
    summary_writer.add_summary(MakeSummary('%s/macro_f1' % mode, maf1), current_step)
    summary_writer.add_summary(MakeSummary('%s/micro_f1' % mode, mif1), current_step)
    summary_writer.add_summary(MakeSummary('%s/exam_f1' % mode, ebf1), current_step)
    summary_writer.add_summary(MakeSummary('%s/acc' % mode, acc), current_step)
    summary_writer.add_summary(MakeSummary('%s/ha' % mode, ha), current_step)

    return nll_loss, best_val_metrics

def main(_):
    print('reading npy...')
    np.random.seed(19940423) # set the random seed of numpy
    data = np.load(FLAGS.data_dir) #load data from the data_dir
    train_idx = np.load(FLAGS.train_idx) #load the indices of the training set
    valid_idx = np.load(FLAGS.valid_idx) #load the indices of the validation set
    test_idx = np.load(FLAGS.test_idx)
    labels = get_data.get_label(data, train_idx) #load the labels of the training set

    print("min:", np.amin(labels))
    print("max:", np.amax(labels))

    print("positive label rate:", np.mean(labels)) #print the rate of the positive labels in the training set
    param_setting = "lr-{}_lr-decay_{:.2f}_lr-times_{:.1f}_nll-{:.2f}_l2-{:.2f}_c-{:.2f}".format(FLAGS.learning_rate, FLAGS.lr_decay_ratio, FLAGS.lr_decay_times, FLAGS.nll_coeff, FLAGS.l2_coeff, FLAGS.c_coeff)
    build_path(FLAGS.summary_dir+param_setting)
    build_path('model/model_{}/{}'.format(FLAGS.dataname, param_setting))

    one_epoch_iter = len(train_idx) / FLAGS.batch_size # compute the number of iterations in each epoch
    print("one_epoch_iter:", one_epoch_iter)

    print('reading completed')
    # config the tensorflow
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=session_config)

    print('showing the parameters...\n')

    # print all the hyper-parameters in the current training
    for key in FLAGS:
        print("%s\t%s"%(key, FLAGS[key].value))
    print()

    print('building network...')

    #building the model 
    hg = model.MODEL(is_training=True)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, global_step, one_epoch_iter * (FLAGS.max_epoch / FLAGS.lr_decay_times), FLAGS.lr_decay_ratio, staircase=True)

    #log the learning rate 
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

    #use the Adam optimizer 
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    reset_optimizer_op = tf.compat.v1.variables_initializer(optimizer.variables())

    #set training update ops/backpropagation
    var_x_encoder = tf.compat.v1.trainable_variables('feat_encoder')
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if FLAGS.resume:
            train_op = optimizer.minimize(hg.total_loss, var_list = var_x_encoder, global_step = global_step)
        else:
            train_op = optimizer.minimize(hg.total_loss, global_step = global_step)

    merged_summary = tf.compat.v1.summary.merge_all() # gather all summary nodes together
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.summary_dir+param_setting+"/", sess.graph) #initialize the summary writer

    sess.run(tf.compat.v1.global_variables_initializer()) # initialize the global variables in tensorflow
    saver = tf.compat.v1.train.Saver(max_to_keep=FLAGS.max_keep) #initializae the model saver

    if FLAGS.saved_ckpt != "":
        saver.restore(sess, FLAGS.saved_ckpt)

    print('building finished')

    #initialize several
    best_loss = 1e10
    best_iter = 0
    best_macro_f1 = 0.
    best_micro_f1 = 0.

    # smooth means average. Every batch has a mean loss value w.r.t. different losses
    smooth_nll_loss=0.0 # label encoder decoder cross entropy loss
    smooth_nll_loss_x=0.0 # feature encoder decoder cross entropy loss
    smooth_l2_loss=0.0 # weights regularization
    smooth_c_loss = 0.0 # label encoder decoder ranking loss
    smooth_c_loss_x=0.0 # feature encoder decoder ranking loss
    smooth_kl_loss = 0.0 # kl divergence
    smooth_total_loss=0.0 # total loss
    smooth_macro_f1 = 0.0 # macro_f1 score
    smooth_micro_f1 = 0.0 # micro_f1 score

    best_macro_f1 = 0.0 # best macro f1 for ckpt selection in validation
    best_micro_f1 = 0.0 # best micro f1 for ckpt selection in validation
    best_acc = 0.0 # best subset acc for ckpt selction in validation

    temp_label=[]
    temp_indiv_prob=[]

    best_test_metrics = None


    # training the model
    for one_epoch in range(FLAGS.max_epoch):
        print('epoch '+str(one_epoch+1)+' starts!')
        np.random.shuffle(train_idx) # random shuffle the training indices

        for i in range(int(len(train_idx)/float(FLAGS.batch_size))):
            start = i*FLAGS.batch_size
            end = (i+1)*FLAGS.batch_size
            input_feat = get_data.get_feat(data,train_idx[start:end]) # get the NLCD features 
            input_label = get_data.get_label(data,train_idx[start:end]) # get the prediction labels 

            #train the model for one step and log the training loss
            indiv_prob, nll_loss, nll_loss_x, l2_loss, c_loss, c_loss_x, kl_loss, total_loss, macro_f1, micro_f1 = train_step(sess, hg, merged_summary, summary_writer, input_label,input_feat, train_op, global_step)
            
            smooth_nll_loss += nll_loss
            smooth_nll_loss_x += nll_loss_x
            smooth_l2_loss += l2_loss
            smooth_c_loss += c_loss
            smooth_c_loss_x += c_loss_x
            smooth_kl_loss += kl_loss
            smooth_total_loss += total_loss
            smooth_macro_f1 += macro_f1
            smooth_micro_f1 += micro_f1
            
            temp_label.append(input_label) #log the labels
            temp_indiv_prob.append(indiv_prob) #log the individual prediction of the probability on each label

            current_step = sess.run(global_step) #get the value of global_step
            lr = sess.run(learning_rate)
            summary_writer.add_summary(MakeSummary('learning_rate', lr), current_step)

            if current_step % FLAGS.check_freq==0: #summarize the current training status and print them out
                nll_loss = smooth_nll_loss / float(FLAGS.check_freq)
                nll_loss_x = smooth_nll_loss_x / float(FLAGS.check_freq)
                l2_loss = smooth_l2_loss / float(FLAGS.check_freq)
                c_loss = smooth_c_loss / float(FLAGS.check_freq)
                c_loss_x = smooth_c_loss_x / float(FLAGS.check_freq)
                kl_loss = smooth_kl_loss / float(FLAGS.check_freq)
                total_loss = smooth_total_loss / float(FLAGS.check_freq)
                macro_f1 = smooth_macro_f1 / float(FLAGS.check_freq)
                micro_f1 = smooth_micro_f1 / float(FLAGS.check_freq)
                
                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob), (-1))
                temp_label = np.reshape(np.array(temp_label), (-1))
                
                temp_indiv_prob = np.reshape(temp_indiv_prob,(-1, FLAGS.label_dim))
                temp_label = np.reshape(temp_label,(-1, FLAGS.label_dim))

                time_str = datetime.datetime.now().isoformat()
                print("step=%d  %s\nlr=%.6f\nmacro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\tl2_loss=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\ntotal_loss=%.6f\n" % (current_step, time_str, lr, macro_f1, micro_f1, nll_loss*FLAGS.nll_coeff, nll_loss_x*FLAGS.nll_coeff, l2_loss*FLAGS.l2_coeff, c_loss*FLAGS.c_coeff, c_loss_x*FLAGS.c_coeff, kl_loss, total_loss))

                temp_indiv_prob=[]
                temp_label=[]

                smooth_nll_loss = 0
                smooth_nll_loss_x = 0
                smooth_l2_loss = 0
                smooth_c_loss = 0
                smooth_c_loss_x = 0
                smooth_kl_loss = 0
                smooth_total_loss = 0
                smooth_macro_f1 = 0
                smooth_micro_f1 = 0

            if current_step % int(one_epoch_iter*FLAGS.save_epoch)==0: #exam the model on validation set
                print("--------------------------------")
                # exam the model on validation set
                current_loss, val_metrics = validation_step(sess, hg, data, merged_summary, summary_writer, valid_idx, global_step, 'val')
                macro_f1, micro_f1 = val_metrics['maF1'], val_metrics['miF1']

                # select the best checkpoint based on some metric on the validation set
                # here we use macro F1 as the selection metric but one can use others
                if val_metrics['maF1'] > best_macro_f1:
                    print('macro_f1:%.6f, micro_f1:%.6f, nll_loss:%.6f, which is better than the previous best one!!!'%(macro_f1, micro_f1, current_loss))

                    best_loss = current_loss
                    best_iter = current_step

                    print('saving model')
                    saved_model_path = saver.save(sess,FLAGS.model_dir+param_setting+'/model',global_step=current_step)
                    print('have saved model to ', saved_model_path)
                    print()

                    if FLAGS.write_to_test_sh:
                        ckptFile = open(FLAGS.test_sh_path.replace('ebird', FLAGS.dataname), "r")
                        command = []
                        for line in ckptFile:
                            arg_lst = line.strip().split(' ')
                            for arg in arg_lst:
                                if 'model/model_{}/lr-'.format(FLAGS.dataname) in arg:
                                    command.append('model/model_{}/{}/model-{}'.format(FLAGS.dataname, param_setting, best_iter))
                                else:
                                    command.append(arg)
                        ckptFile.close()
                        
                        ckptFile = open(FLAGS.test_sh_path.replace('ebird', FLAGS.dataname), "w")
                        ckptFile.write(" ".join(command)+"\n")
                        ckptFile.close()
                best_macro_f1 = max(best_macro_f1, val_metrics['maF1'])
                best_micro_f1 = max(best_micro_f1, val_metrics['miF1'])
                best_acc = max(best_acc, val_metrics['ACC'])
                
                print("--------------------------------")

if __name__=='__main__':
    tf.compat.v1.app.run()

