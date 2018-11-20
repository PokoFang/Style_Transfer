import scipy.io
import numpy as np
import tensorflow as tf
from vgg import *
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import os
import nst_utils as p
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

	layer_style_cost = tf.reduce_sum(tf.square(tf.subtract(GG, GS))) / (4 * (n_c * n_h * n_w) ** 2)
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

def normalize_img(image):
	mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
	image = np.reshape(image, ((1,) + image.shape))
	image = image - mean
	return image

def generate_img(noise_ratio, content_img):
	noise_img = np.random.uniform(-20, 20, content_img.shape).astype('float32')
	generated_img = noise_ratio * noise_img + (1 - noise_ratio) * content_img
	return generated_img

def save_img(file_name, generate_img):
	mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
	generate_img = generate_img + mean
	generate_img = np.clip(generate_img[0], 0, 255).astype('uint8')
	scipy.misc.imsave(file_name, generate_img)


if __name__ == '__main__':
	content_img = scipy.misc.imread('image/content.jpg')
	content_img = normalize_img(content_img) # shape: (1, 1, 300, 400)
	style_img = scipy.misc.imread('image/style_kusamayayoi.jpg')
	style_img = normalize_img(style_img)
	generated_img = generate_img(0.6, content_img) # choose 0.6 as noise ratio
	# plt.imshow(np.clip(generated_img[0], 0.0, 1.0)) # or use: plt.imshow(generated_img[0].astype(np.uint8))
	# plt.show()

	print('Content')
	tf.reset_default_graph()
	sess = tf.InteractiveSession() # 使用預設的 session，不再需要建構 with session 
	# content
	model = load_vgg_model('imagenet-vgg-verydeep-19.mat')
	sess.run(model['input'].assign(content_img))
	out = model['conv4_2'] # choose layer 'conv4_2'
	a_C = sess.run(out)
	a_G = out
	content_cost = cal_content_cost(a_G, a_C)

	# style
	print('Style')
	sess.run(model['input'].assign(style_img))
	style_layers = [('conv1_1', 0.2),
					('conv2_1', 0.2),
					('conv3_1', 0.2),
					('conv4_1', 0.2),
					('conv5_1', 0.2)]
	style_cost = cal_style_cost(model, style_layers) 

	# run optimizer to generate image 
	cost = total_cost(content_cost, style_cost, alpha = 10, beta = 50)
	print('Start Optimizer')
	train_op = tf.train.AdamOptimizer(2.0).minimize(cost)
	iteration = 201
	sess.run(tf.global_variables_initializer())
	sess.run(model['input'].assign(generated_img))
	for i in range(iteration):
		print(i)
		sess.run(train_op)
		generated_img = sess.run(model['input'])

		if i % 50 == 0:
			t_cost, c_cost, s_cost = sess.run([cost, content_cost, style_cost])
			print('Iteration%d: '%i)
			print('Total cost: ', t_cost)
			print('Content cost: ', c_cost)
			print('Style cost: ', s_cost)
			save_img('output/turtle_k%d.png'%i, generated_img)
