import scipy.io
import numpy as np
import tensorflow as tf

img_width = 400
img_height = 300
n_channels = 3

def load_vgg_model(path):
	"""
	vgg_layers[0][i] # i-th layer
	i: 
		0 is conv1_1 (3, 3, 3, 64)
		1 is relu
		2 is conv1_2 (3, 3, 64, 64)
		3 is relu    
		4 is maxpool
		5 is conv2_1 (3, 3, 64, 128)
		6 is relu
		7 is conv2_2 (3, 3, 128, 128)
		8 is relu
		9 is maxpool
		10 is conv3_1 (3, 3, 128, 256)
		11 is relu
		12 is conv3_2 (3, 3, 256, 256)
		13 is relu
		14 is conv3_3 (3, 3, 256, 256)
		15 is relu
		16 is conv3_4 (3, 3, 256, 256)
		17 is relu
		18 is maxpool
		19 is conv4_1 (3, 3, 256, 512)
		20 is relu
		21 is conv4_2 (3, 3, 512, 512)
		22 is relu
		23 is conv4_3 (3, 3, 512, 512)
		24 is relu
		25 is conv4_4 (3, 3, 512, 512)
		26 is relu
		27 is maxpool
		28 is conv5_1 (3, 3, 512, 512)
		29 is relu
		30 is conv5_2 (3, 3, 512, 512)
 		31 is relu
		32 is conv5_3 (3, 3, 512, 512)
		33 is relu
		34 is conv5_4 (3, 3, 512, 512)
		35 is relu
		36 is maxpool

		# for style transfer, fully connected layers will be ignored
		37 is fullyconnected (7, 7, 512, 4096)
		38 is relu
		39 is fullyconnected (1, 1, 4096, 4096)
		40 is relu
		41 is fullyconnected (1, 1, 4096, 1000)
		42 is softmax
	"""
	
	vgg = scipy.io.loadmat(path)
	vgg_layers = vgg['layers']
    
	def _weights(layer):
		layer_name = vgg_layers[0][layer][0][0][2]
		wb = vgg_layers[0][layer][0][0][0]
		W = wb[0][0]
		b = wb[0][1]
		# print(layer_name)
		# print(W)
		# print(b)
		return W, b

	def _conv2d(pre_layer, layer):
		W, b = _weights(layer)
		W = tf.constant(W)
		b = tf.constant(b)
		return tf.nn.conv2d(pre_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

	def _relu(conv2d):
		return tf.nn.relu(conv2d)

	def _conv2d_relu(pre_layer, layer):
		return (_relu(_conv2d(pre_layer, layer)))

	def _avgpool(pre_layer):
		return tf.nn.avg_pool(pre_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
	graph = {}
	graph['input']   = tf.Variable(np.zeros((1, img_height, img_width, n_channels)), dtype = 'float32')
	graph['conv1_1']  = _conv2d_relu(graph['input'], 0)
	graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2)
	graph['avgpool1'] = _avgpool(graph['conv1_2'])
	graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5)
	graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7)
	graph['avgpool2'] = _avgpool(graph['conv2_2'])
	graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10)
	graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12)
	graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14)
	graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16)
	graph['avgpool3'] = _avgpool(graph['conv3_4'])
	graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19)
	graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21)
	graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23)
	graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25)
	graph['avgpool4'] = _avgpool(graph['conv4_4'])
	graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28)
	graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30)
	graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32)
	graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34)
	graph['avgpool5'] = _avgpool(graph['conv5_4'])
	return graph
   
#graph = load_vgg_model('imagenet-vgg-verydeep-19.mat')
#print(graph)