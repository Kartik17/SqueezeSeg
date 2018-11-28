#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Convert csv files into npy files (x, y, z, itensity, depth, lable)
"""

import sys
import os.path
import numpy as np
from PIL import Image
import pandas as pd

# lib_path = os.path.abspath(os.path.join('..'))
# print lib_path
# sys.path.append(lib_path)


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

def hv_in_range(x, y, z, fov, fov_type='h'):
    """
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit

    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`

    Return:
    `cond`:condition of points within fov or not

    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' "
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), \
                                np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), \
                                np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")


def pto_depth_map(velo_points,
                      H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
    """
    Project velodyne points into front view depth map.

    :param velo_points: velodyne points in shape [:,4]
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param C: the channel size of depth map
        3 cartesian coordinates (x; y; z),
        an intensity measurement and
        range r = sqrt(x^2 + y^2 + z^2)
    :param dtheta: the delta theta of H, in radian
    :param dphi: the delta phi of W, in radian
    :return: `depth_map`: the projected depth map of shape[H,W,C]
    """

    x, y, z, i, label = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3], velo_points[:, 4]
    d = np.sqrt(x ** 2 + y ** 2 + z**2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d==0] = 0.000001
    r[r==0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y/r)
    phi_ = (phi/dphi).astype(int)
    phi_[phi_<0] = 0
    phi_[phi_>=512] = 511

    # print(np.min(phi_))
    # print(np.max(phi_))
    #
    # print z
    # print np.radians(2.)
    # print np.arcsin(z/d)
    theta = np.radians(2.) - np.arcsin(z/d)
    # print theta
    theta_ = (theta/dtheta).astype(int)
    # print theta_
    theta_[theta_<0] = 0
    theta_[theta_>=64] = 63
    #print theta,phi,theta_.shape,phi_.shape
    # print(np.min((phi/dphi)),np.max((phi/dphi)))
    #np.savetxt('./dump/'+'phi'+"dump.txt",(phi_).astype(np.float32), fmt="%f")
    #np.savetxt('./dump/'+'phi_'+"dump.txt",(phi/dphi).astype(np.float32), fmt="%f")
    # print(np.min(theta_))
    # print(np.max(theta_))

    depth_map = np.zeros((H, W, C))
    # 5 channels according to paper
    if C == 6:
        depth_map[theta_, phi_, 0] = x
        depth_map[theta_, phi_, 1] = y
        depth_map[theta_, phi_, 2] = z
        depth_map[theta_, phi_, 3] = i
        depth_map[theta_, phi_, 4] = d
        depth_map[theta_, phi_, 5] = label
    else:
        depth_map[theta_, phi_, 0] = i
    return depth_map


def csv2npy(dir):

    points_files_list = os.listdir(os.path.join(dir, 'pts'))
    intensity_files_list = os.listdir(os.path.join(dir, 'intensity'))
    category_files_list = os.listdir(os.path.join(dir, 'category'))

    count = 1
    while count<=50000 :
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

        np_p = np.array(cloud.T)
        # perform fov filter by using hv_in_range
        cond = hv_in_range(x=np_p[:, 0],
                                y=np_p[:, 1],
                                z=np_p[:, 2],
                                fov=[-45, 45])
        # to rotate points according to calibrated points with velo2cam
        # np_p_ranged = np.stack((np_p[:,1],-np_p[:,2],np_p[:,0],np_p[:,3])).T
        np_p_ranged = np_p[cond]

        # get depth map
        lidar = pto_depth_map(velo_points=np_p_ranged, C=6)
        print lidar.shape
        print count, len(x), points_files_list[count]
        np.save('/home/sang/npytest/'+ points_files_list[count]+'.npy',lidar)
        count =count +1
        # publish_image(lidar)

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
    # try:
    #     create_cloud_from_csv('/home/sang/mnt/home/dataset/pcSeg/dataset/training/')
    # except rospy.ROSInterruptException:
    #     pass
    csv2npy('/home/sang/mnt/home/dataset/pcSeg/dataset/training/')