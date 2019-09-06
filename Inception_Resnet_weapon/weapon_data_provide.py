import glob
import json
import os
import pickle
import numpy as np



def split_train_val_sets(images_dir):
    if not os.path.exists(images_dir):
        raise ValueError('`images_dir` does not exist.')
    train_files = {}
    val_files = {}


    class_list = os.listdir(images_dir+"/train") #列出文件夹下所有的目录与文件

    for i in range(0,len(class_list)):
        path = images_dir+"/train/"+class_list[i]+"/img/"
        img_list = os.listdir(path)
        for j in range(0,len(img_list)):
            img_path = os.path.join(path,img_list[j])
            train_files[img_path] = i
    
    class_lists = os.listdir(images_dir+"/validation") #列出文件夹下所有的目录与文件
    for i in range(0,len(class_lists)):
        paths = images_dir+"/validation/"+class_lists[i]
        img_lists = os.listdir(paths)
        for j in range(0,len(img_lists)):
            img_paths = os.path.join(paths,img_lists[j])
            val_files[img_paths] = i
    
    return train_files, val_files




def write_annotation_json(images_dir, train_json_output_path, 
                          val_json_output_path):
    train_files_dict, val_files_dict = split_train_val_sets(images_dir)
    train_json = json.dumps(train_files_dict)
    
    if train_json_output_path.startswith('/data/jinping/weapon_onepixel/datasets'):
        if not os.path.exists('/data/jinping/weapon_onepixel/datasets'):
            os.mkdir('/data/jinping/weapon_onepixel/datasets')
    
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


# if __name__ == '__main__':
#     save_to_jpg('/data/jiaqi/yhc/cifar-10-batches-py/')
