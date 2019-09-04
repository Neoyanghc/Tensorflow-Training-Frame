import glob
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def save_to_jpg(images_dir):
    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    if not os.path.exists(images_dir):
        raise ValueError('`images_dir` does not exist.')
     
    for j in range(1, 6):
        dataName = images_dir +"data_batch_" + str(j)
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")
        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  
            # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            picName = 'data/jiaqi/yhc/cifar/train/'+ str(i + (j - 1)*10000)+'_' + str(Xtr['labels'][i]) + '.jpg'
            plt.imsave(picName, img)
        print(dataName + " loaded.")
    print("test_batch is loading...")
    
    testXtr = unpickle(images_dir+"test_batch")
    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = 'data/jiaqi/yhc/cifar/test/' + str(i)  + '_' + str(testXtr['labels'][i]) +'.jpg'
        plt.imsave(picName, img)
    print("test_batch loaded.")


def split_train_val_sets(images_dir):
    if not os.path.exists(images_dir):
        raise ValueError('`images_dir` does not exist.')
    # image_files 为寻找到的所有的图片路径名
    #  image_files = glob.glob(os.path.join(images_dir, '*.png'))
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))

    # 进行分割
    train_files = []
    val_files = []
    for i in image_files:
        name = i.split('/')[-1]
        if name[1] == "r":
            train_files.append(i)
        else:
            val_files.append(i)
    
    train_dict = _get_labling_dict(train_files)
    val_dict = _get_labling_dict(val_files)
    return train_dict, val_dict

def _get_labling_dict(image_files=None):
    if image_files is None:
        return None
    # cifar 数据的打开方式
    labling_dict = {}
    for image_file in image_files:
        image_name = image_file.split('.')[0]
        labling_dict[image_file] = int(image_name[-1])
    return labling_dict


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


# if __name__ == '__main__':
#     save_to_jpg('/data/jiaqi/yhc/cifar-10-batches-py/')
