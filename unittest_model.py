import unittest
import model
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelTestCase(unittest.TestCase):
	def test_cal_content_cost(self):
		expected = 0.25
		tf.reset_default_graph()
		with tf.Session() as test:
			x = tf.ones([1, 4, 4, 3], dtype=tf.float32)
			y = tf.multiply(tf.ones([1, 4, 4, 3], dtype=tf.float32), 2)
			result = test.run(model.cal_content_cost(x, y))
		self.assertEquals(expected, result)

	def test_gram_matrix(self):
		expected = np.array([[10, 14], [14, 20]])
		tf.reset_default_graph()
		with tf.Session() as test:
			x = tf.constant([[1, 2], [3, 4]])
			result = test.run(model.gram_matrix(x))
		self.assertTrue((expected == result).all())		

if __name__ == '__main__':
	unittest.main(verbosity=2)