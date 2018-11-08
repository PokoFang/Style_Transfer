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
	return layer_style_cost

def cal_style_cost(model, style_layers):
	style_cost = 0
	for layer, coefficent in style_layers:
		out = model[layer]
		a_S = sess.run(out)
		a_G = out
		layer_style_cost = cal_layer_style_cost(a_G, a_S)		
		style_cost += coefficent * layer_style_cost
	return style_cost

def total_cost(content_cost, style_cost, alpha = 10, beta = 40):
	cost = alpha * content_cost + beta * style_cost
	return cost

if __name__ == '__main__':
	content_img = scipy.misc.imread('image/content.jpg')
	style_img = scipy.misc.imread('image/style.jpg')

	model = load_vgg_model('imagenet-vgg-verydeep-19.mat')
	style_layers = [('conv1_1', 0.2),
					('conv2_1', 0.2),
					('conv3_1', 0.2),
					('conv4_1', 0.2),
					('conv5_1', 0.2)]


	sess.run(model["conv4_2"])