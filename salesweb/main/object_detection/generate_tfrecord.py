# -*- coding: utf-8 -*-
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=train_data/labels.csv  --output_path=data/train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=test_data/labels.csv  --output_path=data/test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import pandas as pd
import tensorflow as tf
import json

from PIL import Image

sys.path.append("..")

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
FLAGS = flags.FLAGS

# CSV_PATH = ''
# OUT_PATH = 'main/object_detection/data/train.record'

modelName = sys.argv[1]
CSV_PATH  = 'main/object_detection/images/' + modelName
OUT_PATH  = CSV_PATH + '/train.record'

# TO-DO replace this with label map
def class_text_to_int(row_label):

    f = open(CSV_PATH + "/labelidx.txt", "r")
    data = f.read()
    f.close()
    data = json.loads(data)

    for dt in data:
        if (dt['LABEL'] == row_label):
            print("return:::"+str(dt['IDX']))
            return int(dt['IDX'])

    # if row_label == 'yellow':
    #     return 1
    # elif row_label == 'jordan':
    #     return 2
    # elif row_label == 'koala':
    #     return 3
    # else:
    #     None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    global OUT_PATH
    global CSV_PATH
    try:
        writer = tf.python_io.TFRecordWriter(OUT_PATH)
        path = os.path.join(os.getcwd(), CSV_PATH)
        examples = pd.read_csv(CSV_PATH + '/' + 'train.csv')
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
    
        writer.close()
        output_path = os.path.join(os.getcwd(), OUT_PATH)
#        print('Successfully created the TFRecords: {}'.format(output_path))
        print('True')
    except:
        print('False')
    
if __name__ == '__main__':

    tf.app.run()