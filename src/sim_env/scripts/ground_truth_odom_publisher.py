#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from nav_msgs.msg import Odometry

class GroundTruthOdomPublisher:
    def __init__(self):
        rospy.init_node('ground_truth_odom_publisher')
        
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        rospy.Subscriber('/ground_truth/odom', Odometry, self.ground_truth_callback)
        rospy.loginfo("Ground Truth Odom Publisher initialized")
    
    def ground_truth_callback(self, msg):
        """接收ground truth并发布为odom"""
        odom = Odometry()
        odom.header.stamp = msg.header.stamp  # 使用原始消息时间戳
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        
        odom.pose = msg.pose
        odom.twist = msg.twist
        
        self.odom_pub.publish(odom)
        
        # 发布TF: odom -> base_footprint，使用原始数据时间戳
        self.tf_broadcaster.sendTransform(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
            msg.header.stamp,  # 使用原始消息时间戳，保持同步
            "base_footprint",
            "odom"
        )

if __name__ == '__main__':
    try:
        node = GroundTruthOdomPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass