'''
Train/test image data generator for yolo
Base from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
Read: 
https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
https://www.tensorflow.org/tutorials/load_data/images
https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0
https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras
''' 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

def _get_lines(text_file):
    with open(text_file) as f:
        return f.readlines()


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, annotation_csv, *, shape=(288, 288, 3), batch_size=8, 
        shuffle=True, is_train=True, num_samples=0, max_num_boxes=10, decode_images=True):   
        'Initialization'
        # annotation_csv-> impath,xmin,ymin,xmax,ymax,label... (x, y are normalized)
        # When batch size = 1, expected o/p shape ((h, w, 3), (None, 5))
        self.annotation_csv = annotation_csv
        self.shape = shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.num_samples = num_samples
        self.max_num_boxes = max_num_boxes        
        self.decode_images = decode_images
        self.annot_lines = _get_lines(annotation_csv)
        self.num_samples = len(self.annot_lines) if num_samples == 0 else num_samples       
        self.indices = np.arange(len(self.annot_lines))

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_samples // self.batch_size


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)


    def __getitem__(self, batch):
        'Generate one batch of data'
        batch_indices = self.indices[batch * self.batch_size:
            (batch + 1) * self.batch_size]

        X, Y = [], []
        for i in batch_indices:
            line = self.annot_lines[i]
            x, y = self._parse_single_example(line)
            X.append(x)
            Y.append(y)

        if self.batch_size == 1:
            return X[0], np.array(Y[0], dtype=np.float32)
        else:            
            return np.array(X), np.array(Y, dtype=np.float32)


    def _parse_single_example(self, line):
        subs = line.strip().split(',')
        # y
        entries_per_box = 5
        box_info = subs[1:]
        assert(len(box_info) % entries_per_box == 0)
        # fixed number of boxes
        num_boxes = len(box_info) // entries_per_box        
        if num_boxes > self.max_num_boxes:
            box_info = box_info[: num_boxes * entries_per_box]
        else:
            pad = self.max_num_boxes - num_boxes 
            box_info = box_info + [0] * (entries_per_box * pad)

        y = [float(a) for a in box_info]
        y = np.array(y, dtype=np.float32).reshape(-1, entries_per_box)

        # x
        impath = subs[0]
        if not self.decode_images:
            return (impath, y)
        im = cv2.imread(impath)
        h, w = self.shape[:2]
        im = cv2.resize(im, (w, h))
        # TODO: augmentation in python here, if is_train
        return (im, y)


def preprocess_image(x, y):
    # convert integer types to floats in the [0,1] range.
    x = tf.image.convert_image_dtype(x, tf.float32)
    return (x, y)


def data_generator(annotation_csv, imsize, is_train=True, python_gen_only=False):

    def python_gen():
        im_gen = DataGenerator(annotation_csv, shape=(imsize, imsize, 3),
            batch_size=1, shuffle=is_train, is_train=is_train)
        for xy in im_gen:
            yield xy

    if python_gen_only:
        return python_gen

    dataset = tf.data.Dataset.from_generator(python_gen,   
        output_types=(tf.uint8, tf.float32))
        # output_shapes=(tf.TensorShape([imsize, imsize, 3]), tf.TensorShape([10, 5])))
        # output_shapes is optional?!

    # can be cached if dataset is small
    # dataset = dataset.cache(annotation_csv.replace('.txt', '.tfrecord'))

    dataset = dataset.map(preprocess_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
 
    # no effect?:
    # dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
    return dataset


if __name__ == "__main__":
    pygen = data_generator('../data/voc2012_val.csv', 288, False, True)
    x, y = next(pygen())
    print(x.shape, y)

    tfgen = data_generator('../data/voc2012_val.csv', 288, False)
    for x, y in tfgen.take(2):
        print(x.shape, y)
