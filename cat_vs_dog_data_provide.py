# -*- coding: utf-8 -*-
'''
   provide.py 文件
   主要读入图片数据 将 .jpg 生成json(也可以不这么做)
   return image_file(all path) 和 一个dict['all path':all label]               
'''
import glob
import json
import os


# 传入图片.jpg存放的地址，然后将其分为,train_dict 和 val_dict，dict为[path,label]
def split_train_val_sets(images_dir, val_ratio=0.02):
    if not os.path.exists(images_dir):
        raise ValueError('`images_dir` does not exist.')
    # image_files 为寻找到的所有的图片路径名
    #  image_files = glob.glob(os.path.join(images_dir, '*.png'))
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))

    # 进行分割
    num_val_samples = int(len(image_files) * val_ratio)
    val_files = image_files[:num_val_samples]
    train_files = image_files[num_val_samples:]
    

    train_dict = _get_labling_dict(train_files)
    val_dict = _get_labling_dict(val_files)
    return train_dict, val_dict

# 获取图片对应的label，并返回dict字典
def _get_labling_dict(image_files=None):
    if image_files is None:
        return None
    # kagger，猫vs狗数据集的处理方式
    # 其他数据集，主要修改这里
    labling_dict = {}
    for image_file in image_files:
        image_name = image_file.split('/')[-1]
        if image_name.startswith('cat'):
            labling_dict[image_file] = 0
        elif image_name.startswith('dog'):
            labling_dict[image_file] = 1
    return labling_dict



# 将dict写入json文件中
def write_annotation_json(images_dir, train_json_output_path, 
                          val_json_output_path):
    train_files_dict, val_files_dict = split_train_val_sets(images_dir)
    train_json = json.dumps(train_files_dict)
    
    if train_json_output_path.startswith('./datasets'):
        if not os.path.exists('./datasets'):
            os.mkdir('./datasets')
    
    with open(train_json_output_path, 'w') as writer:
        json.dump(train_json, writer)
    val_json = json.dumps(val_files_dict)
    with open(val_json_output_path, 'w') as writer:
        json.dump(val_json, writer)


# 通过读json文件将对应图片，读入到 new images_dir + images_name 中 return image_file 和 对应的label
def provide(annotation_path=None, images_dir=None):

    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')
        
    with open(annotation_path, 'r') as reader:
        annotation_str = json.load(reader)
        annotation_d = json.loads(annotation_str)

    annotation_dict = {}
    for image_name, labels in annotation_d.items():
        # 是否进行路径拼接
        if images_dir is not None:
            image_name = os.path.join(images_dir, image_name)
            
        annotation_dict[image_name] = labels
    return annotation_dict

