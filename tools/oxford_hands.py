import time
import os
import hashlib
from glob import glob 
from scipy.io import loadmat
import cv2

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import tqdm

flags.DEFINE_string('data_dir', 'GIVE SOME PATH',
                    'path to raw oxford hands dataset')
flags.DEFINE_string('output_file', './data/oxford_hands_xxx.tfrecord', 'output dataset')
flags.DEFINE_string('classes', './data/oxford_hands.names', 'classes file')


def parse_annotation_file(mat_file):
    annot = loadmat(mat_file, squeeze_me=True)
    boxes = annot['boxes']
    if boxes.shape == ():
        boxes = [boxes]    
    ortho_boxes = []
    for box in boxes:
        box = box.item()
        box = [(int(a[1]), int(a[0])) for a in list(box)[:4]]
        a, b, c, d = box
        xs = [p[0] for p in box]
        xmin, xmax = min(xs), max(xs)
        ys = [p[1] for p in box]
        ymin, ymax = min(ys), max(ys)
        ortho_boxes.append([xmin, ymin, xmax, ymax])
    return ortho_boxes


def build_example(image_path, boxes):
    if 0 == len(boxes):
        return None

    img_raw = open(image_path, 'rb').read()
    im = cv2.imread(image_path)
    height, width = im.shape[:2]

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []

    for box in boxes:
        xmin.append(box[0]/width)
        xmax.append(box[2]/width)
        ymin.append(box[1]/height)
        ymax.append(box[3]/height)
        classes_text.append('hand'.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),

        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            image_path.encode('utf8')])),        
    }))
    return example



def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    image_list = glob(os.path.join(FLAGS.data_dir, 'images', '*.jpg'))
    annot_list = glob(os.path.join(FLAGS.data_dir, 'annotations', '*.mat'))
    assert(len(image_list) == len(annot_list))

    image_list= sorted(image_list)
    annot_list= sorted(annot_list)

    logging.info("Image list loaded: %d", len(image_list))

    for image, annot in tqdm.tqdm(zip(image_list, annot_list)):
        boxes = parse_annotation_file(annot)
        tf_example = build_example(image, boxes)
        if tf_example is None:
            continue
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
