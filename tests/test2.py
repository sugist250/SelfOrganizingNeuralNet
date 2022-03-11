# from keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# nomalize_train_images = train_images[:]/255
# print(train_labels[0])
# print(nomalize_train_images[0])
import numpy as np
map_size = 10
input_vec_size = 5
inuts = np.random.uniform(low=-1.0, high=1.0,size=(input_vec_size))
weight = np.random.uniform(low=-1.0, high=1.0,size=(map_size, map_size, input_vec_size))

weight = weight.reshape(map_size*map_size, input_vec_size)
ans = np.dot(weight,inuts)

print(ans.shape)
