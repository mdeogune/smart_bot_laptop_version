import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
#matplotlib inline
import pickle

TRAIN_DIR='/home/mukesh-deo/Documents/dataset_1/train/'
TEST_DIR='/home/mukesh-deo/Documents/dataset_1/test/'
train_image_file_names = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)][0:150] 
test_image_file_names = [TEST_DIR+i for i in os.listdir(TEST_DIR)][0:1]

def decode_image(image_file_names, resize_func=None):
    
    images = []
    
    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file = tf.read_file(file_name)
        image = tf.image.decode_jpeg(file)
        if resize_func != None:
            image = resize_func(image)
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer()
       # tf.initialize_all_variables().run()   
        for i in range(len(image_file_names)):
            images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))
            if (i+1) % 1000 == 0:
                print('Images processed: ',i+1)
        
        session.close()
    
    return images

train_images = decode_image(train_image_file_names)
test_images = decode_image(test_image_file_names)
all_images = train_images + test_images

width = []
height = []
aspect_ratio = []
for image in all_images:
    h, w, d = np.shape(image)
    aspect_ratio.append(float(w) / float(h))
    width.append(w)
    height.append(h)

print('Mean aspect ratio: ',np.mean(aspect_ratio))
plt.plot(aspect_ratio)
plt.show()

print('Mean width:',np.mean(width))
print('Mean height:',np.mean(height))
plt.plot(width, height, '.r')
plt.show()
for i in range(10):
    plt.imshow(train_images[i])
    plt.show()
