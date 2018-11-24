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




def create_cloud_from_csv(csvfile):
    pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)
    rospy.init_node('csv_play', anonymous=True)
    while not rospy.is_shutdown() and csv:
        pf = load(csv)
        x = 
        y = 
        z = 
        i = 
        label = 
        cloud = np.stack((x, y, z, i, label))
        # print len(cloud.T)
        # cloud = np.stack((x, y, z, i))


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
        
        pub.publish(msg_segment)
        rospy.loginfo("info")


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass