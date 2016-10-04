#!/usr/bin/env python

from __future__ import print_function

import h5py
import os
import sys
import numpy as np
import random
import logging

import rospy
from std_msgs.msg import String
from task1_vision_pkg.srv import *

#SAVE_NAME = 'train'

MODEL_PATH_ = '/models/'
PKG_DIR_ = os.getcwd()

## shuffle the training set
def features_from_file_with_shuffle(file_path):
    txtfile = open(file_path, 'r')
    features = []
    for icounter, line in enumerate(txtfile):
        feature = []
        for i, word in enumerate(line.split(' ')):
            feature.append(float(word))
        features.append(feature)
    txtfile.close()

    random.shuffle(features)

    labels = []
    for feature in features:
        lenght = len(feature) - 1
        label = feature.pop(lenght)
        if label == -1:
            label = 0.0
        labels.append(label)
    print(labels)
        
    if len(features) == 0:
        return (None, None)    
    return (np.array(features), np.array(labels))

def features_from_file(file_path):
    txtfile = open(file_path, 'r')
    features = []
    labels = []
    for icounter, line in enumerate(txtfile):
        feature = []
        for i, word in enumerate(line.split(' ')):
            feature.append(float(word))
        lenght = len(feature) - 1
        labels.append(feature.pop(lenght))
        features.append(feature)
    txtfile.close()

    if len(features) == 0:
        return (None, None)    
    return (np.array(features), np.array(labels))

def save_to_hdf5(file_path, features, labels):
    h5_filename = file_path + '.h5'
    with h5py.File(h5_filename, 'w') as f:
        f['data'] = features
        f['label'] = labels

    with open(file_path + '.txt', 'w') as f:
        print(PKG_DIR_ + '/' + h5_filename, file = f)


def caffe_hdf5_convertor_handler(req):
    read_path = req.feature_file_path.data
    
    rospy.logwarn("IN-PATH: %s", read_path)

    features, labels = features_from_file_with_shuffle(read_path)
    
    print(features)
    print(labels)

    status = 1
    if features.shape[0] != labels.shape[0]:
        status = 0
    else:
        save_to_hdf5(req.hdf5_filename.data, features, labels)

    return CaffeHdf5ConvertorResponse(status)

def caffe_hdf5_convertor_server():
    rospy.init_node('caffe_hdf5_convertor_server')
    s = rospy.Service('caffe_hdf5_convertor', 
                      CaffeHdf5Convertor, 
                      caffe_hdf5_convertor_handler)
    rospy.spin()        
        
if __name__ == "__main__":
    caffe_hdf5_convertor_server()
        
