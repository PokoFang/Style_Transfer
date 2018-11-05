import scipy.misc
from PIL import Image

def resize_img(image, out_name):
	img_arr = scipy.misc.imread(image)
	rs_img_arr = scipy.misc.imresize(img_arr, (225, 300))
	rs_img = Image.fromarray(rs_img_arr, 'RGB')
	rs_img.show()
	rs_img.save(out_name)

resize_img('image/monet.jpg', 'image/style.jpg')
resize_img('image/turtle.jpg', 'image/content.jpg')