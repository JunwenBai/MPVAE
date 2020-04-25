import tensorflow as tf
import numpy as np
import config 

FLAGS = tf.app.flags.FLAGS

def get_label(data, order):
	output = []
	offset = FLAGS.meta_offset
	for i in order:
		output.append(data[i][offset:(offset+FLAGS.label_dim)])
	output = np.array(output, dtype="int") 
	return output

def get_feat(data, order, offset = None):
	output = []
	if (offset == None):
		offset = FLAGS.meta_offset + FLAGS.label_dim
	for i in order:
		output.append(data[i][offset:offset + FLAGS.feat_dim])
	output = np.array(output, dtype="float32") 
	return output

