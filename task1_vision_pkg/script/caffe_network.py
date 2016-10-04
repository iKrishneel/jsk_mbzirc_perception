#!/usr/bin/env python

import numpy as np
import sys
import os
import h5py

import rospy
from std_msgs.msg import String, Int64
from task1_vision_pkg.srv import *


CAFFE_ROOT = '../caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')

import caffe
from caffe import layers as L
from pylab import *

def debug_training(solver):
    NUM_OUTPUT_ = 2
    deb_iter = 200
    deb_test_interval = 10
    deb_train_loss = zeros(deb_iter)
    deb_test_acc = zeros(int(np.ceil(deb_iter * 1.0 / deb_test_interval)))
    output = zeros((deb_iter, 8, NUM_OUTPUT_))

    for it in range(deb_iter):
        solver.step(1)
        deb_train_loss[it] = solver.net.blobs['loss'].data
        solver.test_nets[0].forward(start='data')
        output[it] = solver.test_nets[0].blobs['fc3'].data[:8]

        if it % deb_test_interval == 0:
            print "\033[34m Test Iteration \033[0m", it
            correct = 0
            data = solver.test_nets[0].blobs['fc3'].data
            label = solver.test_nets[0].blobs['label'].data

            for test_it in range(100):
                solver.test_nets[0].forward()

                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if data[i][j] > 0 and label[i] == 1:
                            correct += 1
                        elif data[i][j] <=0 and label[i] == -1:
                            correct += 1
            deb_test_acc[int(it / deb_test_interval)] = correct * 1.0 / (len(data) * len(data[0]) * 100)
            
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(deb_iter), deb_train_loss)
    ax2.plot(deb_test_interval * arange(len(deb_test_acc)), deb_test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    _.savefig('converge.png')

def get_features(file_path):
    txtfile = open(file_path, 'r')
    features = []
    labels = []
    for icounter, line in enumerate(txtfile):
        feature = []
        for word in line.split(' '):
            feature.append(float(word))
        lenght = len(feature) - 1
        labels.append(feature.pop(lenght))
        features.append(feature)
    txtfile.close()
    return (np.array(features), np.array(labels))
    
def test_classifier(caffe_model, proto_file):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(proto_file, caffe_model, caffe.TEST)
    feature_vector, labels = get_features('features/test_features.txt')
    indx = 2
    print feature_vector[indx].shape
    
    for feature, label in zip(feature_vector, labels):
        #print feature
        net.blobs['data'].data[...] = feature
        output = net.forward()
        #print "\033[34m ", output, "\033[0m "
        output_prob = output['loss'][0]
        print output
        print "\033[34m ", output_prob.argmax(), ", ", label, "\033[0m "

    
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)        


def train_classifer(solver_path):
    path_to_hdf5 = 'train.txt'

    caffe.set_device(0)
    caffe.set_mode_gpu()

    solver = caffe.SGDSolver(solver_path)
    solver.net.forward()

    debug_net = False
    if debug_net:
        solver.test_nets[0].forward()
        solver.step(1)

        debug_training(solver)
    else:
        solver.solve()

    print solver.test_nets[0].blobs['fc3'].data
    print solver.test_nets[0].blobs['label'].data
    

def caffe_network_handler(req):
    train_classifer(req.caffe_solver_path.data)
    status = Int64()
    status.data = 1
    return CaffeNetworkResponse(status)

        
def caffe_network_server():
    rospy.init_node('caffe_network_server')
    s = rospy.Service('caffe_network',
                      CaffeNetwork,
                      caffe_network_handler)
    rospy.spin()    

if __name__ == "__main__":
    caffe_network_server()

