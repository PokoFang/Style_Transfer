import scipy.io
import numpy as np
import tensorflow as tf
import vgg
import scipy.misc
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

content_image = scipy.misc.imread('image/content.jpg')

def cal_content_cost(a_G, a_C):
	# a_G.shape returns a tensor of shape, and needs to run sess to get the value
	# a_G.get_shape() returns a tuple
	m, n_h, n_w, n_c = a_G.get_shape().as_list()
	J_content = tf.reduce_sum(tf.square(tf.subtract(a_G, a_C))) / (4 * n_h * n_w * n_c)
	return J_content
