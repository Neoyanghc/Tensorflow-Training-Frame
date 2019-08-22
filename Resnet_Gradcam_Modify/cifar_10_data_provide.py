import glob
import json
import os

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
        dataName = "data_batch_" + str(j) 
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")
        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  
            # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            picName = images_dir+'/train/'+str(Xtr['labels'][i])
            \							+ '_' + str(i + (j - 1)*10000) + '.jpg'  
            imsave(picName, img)
        print(dataName + " loaded.")
    print("test_batch is loading...")
    
    testXtr = unpickle("test_batch")
    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = images_dir+'/test/' + str(testXtr['labels'][i]) + '_'+ str(i)'.jpg'
        imsave(picName, img)
    print("test_batch loaded.")
    
def _get_labling_dict_of_cifar(image_files=None):
    if image_files is None:
        return None
    # cifar 数据的打开方式
    labling_dict = {}
    for image_file in image_files:
        image_name = image_file.split('/')[-1]
        labling_dict[image_file] = image_name[0]
    return labling_dict
