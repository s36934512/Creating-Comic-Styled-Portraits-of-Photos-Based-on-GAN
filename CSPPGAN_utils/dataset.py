import tensorflow as tf
import numpy as np
import random
from pathlib import Path

def load_and_preprocess_from_path(path, input_shape, sketch=None):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=input_shape[2], expand_animations=False)
    image = tf.image.resize(image, [input_shape[0], input_shape[1]])

    if sketch:
        Y = 0.3 * image[:,:, 0] + 0.59 * image[:,:, 1] + 0.11 * image[:,:, 2]
        Y = tf.expand_dims(Y, 2)
        Y = tf.expand_dims(Y, 0)
        max_pooled = 255 - tf.nn.pool(Y, window_shape=(3, 3), pooling_type='MAX', padding='SAME')

        Y = Y + (Y * max_pooled) / (255 - max_pooled + 1e-5)
        Y = tf.squeeze(Y, axis=0)
        image = tf.concat([Y,Y], 2)
        image = tf.concat([image,Y], 2)
        
    image = tf.cast(image, tf.float32) / 255.0
    return image

def dataset_from_path_list(path_list, size, sketch=None):
    batch_size = size[0]
    image_size = size[1:]

    dataset = tf.data.Dataset.from_tensor_slices(path_list).map(lambda x: load_and_preprocess_from_path(x, image_size, sketch))
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1).cache()
    return dataset

def dataset_from_directrory(directrory, size, num=None, shuffle=True, random_seed=13, sketch=None):
    img_paths = sorted([str(img_path) for img_path in Path(directrory).iterdir()])
   
    if num:
        img_paths = img_paths[:num]

    if shuffle:
        random.Random(random_seed).shuffle(img_paths)
        
    dataset = dataset_from_path_list(img_paths, size, sketch)
    return dataset


def dataset_from_multiple_directrory(directrory_list, size, num=None, shuffle=True, random_seed=13, sketch_list=None):
    dataset_list = []
    
    if sketch_list == None:
        sketch_list = [False for i in directrory_list]
    
    for directrory, sketch in zip(directrory_list, sketch_list):
        dataset_list.append(dataset_from_directrory(directrory, size, num=num, shuffle=shuffle,
                                                    random_seed=random_seed, sketch=sketch))

    n_batchs = min([len(x) for x in dataset_list])
    return n_batchs, dataset_list
