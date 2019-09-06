#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
generate_tfrecord.py

利用data_provide.py 传递出来的参数进行将images.jpg 变成 .record 文件

利用 tf.dataset API 可以进行处理

指定 图片的路径，然后 图片resize的固定大小
"""

import io
import tensorflow as tf
import time
from PIL import Image
from PIL import ImageFile
# import data_provider 引入对应的data provide 文件
from weapon_data_provide import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 对应参数 说明
flags = tf.app.flags
# TODO
flags.DEFINE_string('images_dir', 
                    '/data/jinping/weapon_onepixel',
                    'Path to images (directory).')
flags.DEFINE_string('train_annotation_path', 
                    '/data/jinping/weapon_onepixel/datasets/train.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('train_output_path', 
                    '/data/jinping/weapon_onepixel/datasets/train.record',
                    'Path to output tfrecord file.')
flags.DEFINE_string('val_annotation_path', 
                    '/data/jinping/weapon_onepixel/datasets/val.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('val_output_path', 
                    '/data/jinping/weapon_onepixel/datasets/val.record',
                    'Path to output tfrecord file.')
# TODO
flags.DEFINE_integer('resize_side_size',299, 'Resize images to fixed size.')
FLAGS = flags.FLAGS



# tfrecord 文件的多类型读取
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



# 定义协议和对一个图片数据进行 创建
def create_tf_example(image_path, label, resize_size=None):
    
    # 打开图片文件 python open 函数也可以，但是tf.gfile 可以打开tf的指定文件
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    
    # 在内存中读取IO文件
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    width, height = image.size
    
    # Resize
    if resize_size is not None:
        if width > height:
            width = int(width * resize_size / height)
            height = resize_size
        else:
            width = resize_size
            height = int(height * resize_size / width)
        image = image.resize((width, height), Image.ANTIALIAS)
        image = image.convert('RGB')
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()
    
    # 定义tfrecord的协议
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpg'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)}))
    return tf_example


# 利用输入的json文件中的dict进行tfrecord文件的写出
def generate_tfrecord(annotation_dict, output_path, resize_size=None):
    num_valid_tf_example = 0
    # 创建一个write的实例对象
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print('%s does not exist.' % image_path)
            continue
        tf_example = create_tf_example(image_path, label, resize_size)
        writer.write(tf_example.SerializeToString())
        num_valid_tf_example += 1
        
        # 输出创建信息进行显示
        if num_valid_tf_example % 1000 == 0:
            print('Create %d TF_Example.' % num_valid_tf_example)
    writer.close()
    print('Total create TF_Example: %d' % num_valid_tf_example)
    
    
def main(_):
    images_dir = FLAGS.images_dir
    train_annotation_path = FLAGS.train_annotation_path
    train_record_path = FLAGS.train_output_path
    val_annotation_path = FLAGS.val_annotation_path
    val_record_path = FLAGS.val_output_path
    resize_size = FLAGS.resize_side_size
    
    # Write json
    write_annotation_json(images_dir, train_annotation_path,val_annotation_path)
    
    time.sleep(5)
    
    # 获得图片的地址映射json 文件
    train_annotation_dict = provide(train_annotation_path, None)
    val_annotation_dict = provide(val_annotation_path, None)
    
    # 分别形成对应的tfrecord,存储在对应的文件中
    generate_tfrecord(train_annotation_dict, train_record_path, resize_size)
    generate_tfrecord(val_annotation_dict, val_record_path, resize_size)
    
    
if __name__ == '__main__':
    tf.app.run()