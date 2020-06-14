from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.models import YoloV3, YoloV3Tiny, YoloV2Lite
from yolov3_tf2.utils import load_darknet_weights
import tensorflow as tf
import cv2
import os

from convert_to_pb import convert_to_pb

tf_version = tf.__version__.split('.')[0]

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_boolean('from_h5', False, 'source of weights')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('lite', False, 'yolov3 or yolov2-lite')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('size', 0, 'Default is None')
flags.DEFINE_boolean('training', False, 'training flag')


def predict(model):
    size = model.input_shape[1]
    if size is None:
        size = 288
    im = cv2.imread('data\meme.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size))
    im = im / 255.
    im = np.expand_dims(im, axis=0)

    p = model.predict(im)
    p = p[0]
    filename = os.path.join('unversion', os.path.basename(model.name) + '_pred.txt')
    print('writing predictions to ' + filename)
    with open(filename, 'w') as f:
        f.write(str(p))    



def main(_argv):
    if tf_version == '2':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    size = None if FLAGS.size == 0 else FLAGS.size

    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=size, classes=FLAGS.num_classes, training=FLAGS.training)
    elif FLAGS.lite:
        yolo = YoloV2Lite(size=size, classes=FLAGS.num_classes, training=FLAGS.training)
    else:
        yolo = YoloV3(size=size, classes=FLAGS.num_classes, training=FLAGS.training)

    yolo.summary()
    logging.info('model created')

    if FLAGS.from_h5:
    	yolo.load_weights(FLAGS.weights)
    else:
        load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny, FLAGS.lite)
        logging.info('weights loaded')
        
    imsize = 320 if size is None else size
    img = np.random.random((1, imsize, imsize, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    predict(yolo)
    
    if FLAGS.output.endswith(('.h5', '.hdf5')):
        yolo.save(FLAGS.output)
        logging.info('keras model saved')
    else:
        yolo.save_weights(FLAGS.output)
        logging.info('weights saved')

    if tf_version == '1':       
        convert_to_pb(yolo)
        logging.info('pb saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
