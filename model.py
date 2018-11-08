import scipy.io
import numpy as np
import tensorflow as tf
import vgg
import scipy.misc
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cal_content_cost(a_G, a_C):
	# a_G.shape returns a tensor of shape, and needs to run sess to get the value
	# a_G.get_shape() returns a tuple
	m, n_h, n_w, n_c = a_G.get_shape().as_list()
	content_cost = tf.reduce_sum(tf.square(tf.subtract(a_G, a_C))) / (4 * n_h * n_w * n_c)
	return content_cost
	
def gram_matrix(matrix):
	matrix_T = tf.matrix_transpose(matrix)
	G_matrix = tf.matmul(matrix, matrix_T)
	return G_matrix

def cal_layer_style_cost(a_G, a_S): # For a specific layer, the similarity between each channel's activative values
	m, n_h, n_w, n_c = a_G.get_shape().as_list()
	a_G = tf.reshape(tf.transpose(a_G), [n_c, -1]) # flatten 4d matrix to 2d matrix
	a_S = tf.reshape(tf.transpose(a_S), [n_c, -1])

	GG = gram_matrix(a_G) # gram matrix of generated matrix
	GS = gram_matrix(a_S) 

	style_cost = tf.reduce_sum(tf.square(tf.subtract(GG, GS))) / (4 * (n_c * n_h * n_w) ** 2)
	return style_cost


if __name__ == '__main__':
	content_img = scipy.misc.imread('image/content.jpg')
	style_img = scipy.misc.imread('image/style.jpg')
