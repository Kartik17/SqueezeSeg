#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Visualize the training npy files in form of segmented pointcloud
"""

import argparse
import tensorflow as tf
import rospy

from nodes.npy2cloud import SegmentNode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', './data/SqueezeSeg/model.ckpt-23000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/samples/*',
    """Input lidar scan to be detected. Can process glob input such as """
    """./data/samples/*.npy or single input.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/samples_out/', """Directory to dump output.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string('npy_dir', '/velodyne_points', """sub topic""")
tf.app.flags.DEFINE_integer('is_ground_truth', 1, """whether show ground truth label""")


if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud semantic segmentation')
    parser.add_argument('--npy_dir', type=str,
                        help='the npy files dir, default `/home/sang/mnt/home/dataset/pcSeg/lidar_2d/`',
                        default='/home/sang/mnt/home/dataset/pcSeg/lidar_2d/')
    parser.add_argument('--is_ground_truth', type=int,
                        help='0 for predicted label, 1 for ground truth label`',
                        default=1)
    parser.add_argument('--sub_topic', type=str,
                        help='the pointcloud message topic to be subscribed, default `/kitti/points_raw`',
                        default='/kitti/points_raw')
    parser.add_argument('--pub_topic', type=str,
                        help='the pointcloud message topic to be published, default `/squeeze_seg/points`',
                        default='/squeeze_seg/points')
    parser.add_argument('--pub_feature_map_topic', type=str,
                        help='the 2D spherical surface image message topic to be published, default `/squeeze_seg/feature_map`',
                        default='/squeeze_seg/feature_map')
    parser.add_argument('--pub_label_map_topic', type=str,
                        help='the corresponding ground truth label image message topic to be published, default `/squeeze_seg/label_map`',
                        default='/squeeze_seg/label_map')
    args = parser.parse_args()

    rospy.init_node('segment_node')
    node = SegmentNode(is_ground_truth=args.is_ground_truth,
                       npy_dir=args.npy_dir,
                       pub_topic=args.pub_topic,
                       pub_feature_map_topic=args.pub_feature_map_topic,
                       pub_label_map_topic=args.pub_label_map_topic,
                       FLAGS=FLAGS)

    rospy.logwarn("finished.")