#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Node class for publish and segment pointcloud from npy files
"""

import sys
import os.path
import numpy as np
from PIL import Image
import pandas as pd

# lib_path = os.path.abspath(os.path.join('..'))
# print lib_path
# sys.path.append(lib_path)
import tensorflow as tf

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

sys.path.append("..")
from config import *
from nets import SqueezeSeg
from utils.util import *
from utils.clock import Clock




def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

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

class ImageConverter(object):
    """
    Convert images/compressedimages to and from ROS

    From: https://github.com/CURG-archive/ros_rsvp
    """

    _ENCODINGMAP_PY_TO_ROS = {'L': 'mono8', 'RGB': 'rgb8',
                              'RGBA': 'rgba8', 'YCbCr': 'yuv422'}
    _ENCODINGMAP_ROS_TO_PY = {'mono8': 'L', 'rgb8': 'RGB',
                              'rgba8': 'RGBA', 'yuv422': 'YCbCr'}
    _PIL_MODE_CHANNELS = {'L': 1, 'RGB': 3, 'RGBA': 4, 'YCbCr': 3}

    @staticmethod
    def to_ros(img):
        """
        Convert a PIL/pygame image to a ROS compatible message (sensor_msgs.Image).
        """

        # Everything ok, convert PIL.Image to ROS and return it
        if img.mode == 'P':
            img = img.convert('RGB')

        rosimage = ImageMsg()
        rosimage.encoding = ImageConverter._ENCODINGMAP_PY_TO_ROS[img.mode]
        (rosimage.width, rosimage.height) = img.size
        rosimage.step = (ImageConverter._PIL_MODE_CHANNELS[img.mode]
                         * rosimage.width)
        rosimage.data = img.tobytes()
        return rosimage

class SegmentNode():
    """LiDAR point cloud segment ros node"""

    def __init__(self,
                 is_ground_truth, npy_dir, pub_topic, pub_feature_map_topic, pub_label_map_topic,
                 FLAGS):
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        # self._mc = kitti_squeezeSeg_config()
        self._mc = alib_squeezeSeg_config()
        self._mc.LOAD_PRETRAINED_MODEL = False
        # TODO(bichen): fix this hard-coded batch size.
        self._mc.BATCH_SIZE = 1
        self._model = SqueezeSeg(self._mc)
        self._saver = tf.train.Saver(self._model.model_params)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)

        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)
        self._feature_map_pub = rospy.Publisher(pub_feature_map_topic, ImageMsg, queue_size=1)
        self._label_map_pub = rospy.Publisher(pub_label_map_topic, ImageMsg, queue_size=1)
        # self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        # self.use_top_k = rospy.get_param('~use_top_k', 5)

        # rospy.spin()
        self.npy2cloud(is_ground_truth, npy_dir)

    def npy2cloud(self, is_ground_truth, npy_dir):
        """

        :param cloud_msg:
        :return:
        """
        if is_ground_truth:
            rospy.loginfo("SHOW GROUND TRUTH LABEL !")
        else:
            rospy.loginfo("SHOW PREDICTED LABEL !")

        files_list = os.listdir(npy_dir)
        files_list.sort()
        count = 0
        while not rospy.is_shutdown() and count<len(files_list):
            clock = Clock()
            npy_file = files_list[count]
            
            points = np.load(os.path.join(npy_dir, npy_file))
            lidar = points[:,:,:5]
            lidar_f = lidar.astype(np.float32)
            #normalize intensity from [0,255] to [0,1], as shown in KITTI dataset
            #dep_map[:,:,0] = (dep_map[:,:,0]-0)/np.max(dep_map[:,:,0])
            #dep_map = cv2.resize(src=dep_map,dsize=(512,64))

            # to perform prediction
            lidar_mask = np.reshape(
                (lidar[:, :, 4] > 0),
                [self._mc.ZENITH_LEVEL, self._mc.AZIMUTH_LEVEL, 1]
            )
            lidar_f = (lidar_f - self._mc.INPUT_MEAN) / self._mc.INPUT_STD
            pred_cls = self._session.run(
                self._model.pred_cls,
                feed_dict={
                    self._model.lidar_input: [lidar_f],
                    self._model.keep_prob: 1.0,
                    self._model.lidar_mask: [lidar_mask]
                }
            )
            label = pred_cls[0]
            if is_ground_truth:
                label = points[:, :, 5]
            # df = pd.DataFrame(label)
            # print df.size
            # df.to_csv("/home/sang/categray.csv")
            

            # # generated depth map from LiDAR data
            depth_map = Image.fromarray(
                (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
            
            label_3d = np.zeros((label.shape[0], label.shape[1], 3))
            label_3d[np.where(label==0)] = [1., 1., 1.]
            label_3d[np.where(label==1)] = [0., 1., 0.]
            label_3d[np.where(label==2)] = [1., 1., 0.]
            label_3d[np.where(label==3)] = [0., 1., 1.]

            
            x = lidar[:, :, 0].reshape(-1)
            y = lidar[:, :, 1].reshape(-1)
            z = lidar[:, :, 2].reshape(-1)
            i = lidar[:, :, 3].reshape(-1)
            label = label.reshape(-1)
            # print len(label)
            # cond = (label!=0)
            # print(cond)
            cloud = np.stack((x, y, z, i, label))
            print len(cloud.T)
            # cloud = np.stack((x, y, z, i))

            label_map = Image.fromarray(
                (255 * _normalize(label_3d)).astype(np.uint8))

            header = Header()
            header.stamp = rospy.Time()
            header.frame_id = "velodyne"
            # feature map & label map
            msg_feature = ImageConverter.to_ros(depth_map)
            msg_feature.header = header
            msg_label = ImageConverter.to_ros(label_map)
            msg_label.header = header

            # point cloud segments
            # 4 PointFields as channel description
            msg_segment = pc2.create_cloud(header=header,
                                        fields=_make_point_field(cloud.shape[0]),
                                        points=cloud.T)

            self._feature_map_pub.publish(msg_feature)
            self._label_map_pub.publish(msg_label)
            self._pub.publish(msg_segment)
            rospy.loginfo("Point cloud %d processed. Took %.6f ms.",count, clock.takeRealTime())
            count = count+1

    