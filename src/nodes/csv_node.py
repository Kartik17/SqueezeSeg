#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Load and publish pointcloud2 from traing dataset csv files
"""

import sys
import os.path
import numpy as np
from PIL import Image
import pandas as pd
import argparse

# lib_path = os.path.abspath(os.path.join('..'))
# print lib_path
# sys.path.append(lib_path)
import tensorflow as tf

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

# sys.path.append("..")
# from config import *
# from nets import SqueezeSeg
# from utils.util import *
# from utils.clock import Clock

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]     




def create_cloud_from_csv(dir):
    pub = rospy.Publisher("csv_clouds", PointCloud2, queue_size=1)
    rospy.init_node('csv_play', anonymous=True)
    
    points_files_list = os.listdir(os.path.join(dir, 'pts'))
    intensity_files_list = os.listdir(os.path.join(dir, 'intensity'))
    category_files_list = os.listdir(os.path.join(dir, 'category'))

    count = 1
    while not rospy.is_shutdown() and count<=5000 :
        points_df = pd.read_csv(os.path.join(dir,'pts',points_files_list[count]), sep=',')
        intensity_df = pd.read_csv(os.path.join(dir, 'intensity',intensity_files_list[count]))
        label_df = pd.read_csv(os.path.join(dir, 'category', category_files_list[count]))
        x = points_df.values[:,0]
        y = points_df.values[:,1]
        z = points_df.values[:,2]
        i = intensity_df.values[:,0]
        # print i
        label = label_df.values[:,0]
        # print label
        cloud = np.stack((x, y, z, i, label))
        # print len(cloud.T)
        # cloud = np.stack((x, y, z, i))


        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"
        # feature map & label map

        # point cloud segments
        # 4 PointFields as channel description
        msg_segment = pc2.create_cloud(header=header,
                                    fields=_make_point_field(cloud.shape[0]),
                                    points=cloud.T)
        
        pub.publish(msg_segment)
        rospy.loginfo("%d cloud: %d points loaded", count, len(x) )
        count = count + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load and publish pointcloud from csv files ')
    parser.add_argument('--dataset_path', type=str,
                        help='the path of csv files, default `/home/sang/mnt/home/dataset/pcSeg/dataset/training/`',
                        default='/home/sang/mnt/home/dataset/pcSeg/dataset/training/')
    args = parser.parse_args()
    try:
        create_cloud_from_csv(args.dataset_path)
    except rospy.ROSInterruptException:
        pass