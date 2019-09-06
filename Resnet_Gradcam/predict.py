# -*- coding: utf-8 -*-


import cv2
import glob
import os
import tensorflow as tf
import predictor
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('frozen_inference_graph_path',
                    './training/frozen_inference_graph_pb/'+
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', 
                    './test_image', 
                    'Path to images (directory).')
FLAGS = flags.FLAGS


if __name__ == '__main__':
    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
    images_dir = FLAGS.images_dir
    pre_model = predictor.Predictor(frozen_inference_graph_path)
    
    image_files = glob.glob(os.path.join(images_dir, '*.*'))

    true_label = []
    val_label = []
    predicted_count = 0
    num_samples = len(image_files)
    acc = 0
    tse = []
    for image_path in image_files:
        predicted_count += 1

        
        image_name = image_path.split('/')[-1]

        image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        true_label.append(image_name)

        pred_label, top_conv, norm_grads_cam= pre_model.predict([image])

        val_label.append(pred_label)
        if pred_label == image_name:
            acc += 1
        if predicted_count % 100 == 0:
            print('Predict {}.'.format(acc/predicted_count))
        
        tse.append(np.mean(top_conv, axis=0))

        # cam = pre_model.grad_cam(top_conv, norm_grads_cam)
        # pre_model.generate_GradCAM_Image(save_dir='./Grad_CAM_Split/',
        #                                 single_img = image ,
        #                                 cam=cam,
        #                                 save_name="Gram_cam_"+image_name)
        # print('Image Name: %s' % image_name,'Pred Label: %d' % pred_label[0])
    
