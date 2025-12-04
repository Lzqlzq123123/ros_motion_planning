#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import Transform, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


class TFMap2Odom:
    def __init__(self) -> None:
        self.odom_frame_id = rospy.get_param("~odom_frame_id", "odom")

        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Timer(rospy.Duration(0.02), self.timer_callback)  # 50Hz，提高更新频率满足导航需求

        # 静态变换，只在初始化时设置一次
        self.tf = TransformStamped()
        self.tf.header.frame_id = "map"
        self.tf.child_frame_id = self.odom_frame_id
        self.tf.transform.translation.x = 0.0
        self.tf.transform.translation.y = 0.0
        self.tf.transform.translation.z = 0.0
        self.tf.transform.rotation.w = 1.0
        self.tf.transform.rotation.x = 0.0
        self.tf.transform.rotation.y = 0.0
        self.tf.transform.rotation.z = 0.0

    def timer_callback(self, event) -> None:
        # 使用当前ROS时间并稍微超前，给TF查询留出缓冲时间
        self.tf.header.stamp = rospy.Time.now() + rospy.Duration(0.05)
        self.tf_pub.sendTransform(self.tf)


# Start the node
if __name__ == "__main__":
    rospy.init_node("tf_map2odom_node")

    node = TFMap2Odom()

    rospy.spin()
