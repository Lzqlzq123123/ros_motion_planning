#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Navigation data collector for diffusion policy training.

功能：
- 起点由现有仿真/launch 决定，不做随机重置。
- 每个 episode 随机采样一个可达目标点（基于 /map OccupancyGrid）。
- 按照 user_config + move_base 的传统规控导航到目标。
- 在导航过程中以固定频率记录 (obs, action)：
  * 观测：机器人位姿 (x,y,yaw)、线速度/角速度 (v,omega)、目标相对位置 (x_rel,y_rel)、上一时刻动作、RGB 图像
  * 动作：差速小车命令 (v_cmd, omega_cmd)
- 每个 episode 保存为一个 HDF5 文件：episode_xxxxxx.h5

数据格式：HDF5，包含：
- poses, twists, goal_rels, prev_actions, actions, dones, images
- start_pose, goal_pose

关键改进：
- 不使用 cv_bridge（避免 libp11-kit/libffi 动态库冲突）
- 直接从 ROS Image 消息的字节流提取图像数据
- 使用纯 NumPy 进行图像处理和 resize
"""

import os
import math
import yaml
import numpy as np
import threading

import rospy

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseActionGoal

import rosbag
from datetime import datetime

class NavDataCollector(object):
    def __init__(self):
        rospy.init_node("nav_data_collector")
        self.lock = threading.RLock()

        # === 1. 读取配置 ===
        default_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "user_config", "data_collection.yaml",
        )
        config_path = rospy.get_param("~config_path", default_cfg)
        rospy.loginfo("[NavDataCollector] Using config: %s", config_path)

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.data_root = self.cfg.get("data_root", "/tmp/nav_diffusion_data")
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.episode_max_steps = int(self.cfg.get("episode_max_steps", 500))
        self.episode_timeout = float(self.cfg.get("episode_timeout_sec", 120.0))
        self.record_rate = float(self.cfg.get("record_rate_hz", 10.0))

        # 话题配置
        topics = self.cfg.get("topics", {})
        prefix = topics.get("prefix", "")
        self.topic_cmd = prefix + topics.get("cmd_vel", "/cmd_vel")
        self.topic_img = prefix + topics.get("rgb_image", "/camera/rgb/image_raw")
        self.topic_map = prefix + topics.get("map", "/map")
        self.topic_goal = topics.get("goal_pub", "/move_base/goal")
        self.topic_ground_truth = prefix + topics.get("ground_truth", "/ground_truth/state")


        # 坐标系
        frames = self.cfg.get("frames", {})
        self.map_frame = frames.get("map", "map")
        self.base_frame = frames.get("base", "base_link")

        # 图像配置
        img_cfg = self.cfg.get("image", {})
        self.img_w = int(img_cfg.get("width"))
        self.img_h = int(img_cfg.get("height"))

        # 目标模式：random / record_existing
        self.goal_mode = self.cfg.get("goal_mode", "random")
        if self.goal_mode not in ["random", "record_existing"]:
            rospy.logwarn("[NavDataCollector] Unknown goal_mode '%s', fallback to 'random'", self.goal_mode)
            self.goal_mode = "random"

        # record_existing模式需要监听正在执行的目标
        self.current_goal = None  # 当前正在执行的目标 (x, y, yaw)
        self.is_recording = False  # 是否正在记录数据
        self.episode_data = None  # 当前episode的数据缓存
        self.start_pose = None  # episode起始位姿
        self.episode_id = 0  # episode编号
        self.current_bag = None
        self.current_bag_path = None
        self.bag_message_count = 0
        self.last_episode_summary = None
        self.manual_goal = None
        self.manual_goal_available = False

        # === 2. 订阅话题 ===
        self.cmd = None
        self.image = None
        self.map_msg = None
        self.ground_truth = None  # Odometry message from p3d plugin

        rospy.Subscriber(self.topic_cmd, Twist, self.cmd_cb, queue_size=1)
        rospy.Subscriber(self.topic_img, Image, self.image_cb, queue_size=1)
        rospy.Subscriber(self.topic_map, OccupancyGrid, self.map_cb, queue_size=1)
        rospy.Subscriber(self.topic_ground_truth, Odometry, self.ground_truth_cb, queue_size=1)

        # 在record_existing模式下订阅/move_base/goal监听正在执行的目标
        if self.goal_mode == "record_existing":
            self.goal_sub = rospy.Subscriber(self.topic_goal, MoveBaseActionGoal, self.exec_goal_cb, queue_size=1)
            rospy.loginfo("[NavDataCollector] In record_existing mode, will listen for goals on %s", self.topic_goal)

        rospy.loginfo("[NavDataCollector] Goal mode: %s", self.goal_mode)

        # MoveBase Action Goal Publisher
        self.goal_pub = rospy.Publisher(self.topic_goal, MoveBaseActionGoal, queue_size=1)
        self.next_goal_id = 1

        rospy.loginfo("[NavDataCollector] Waiting for basic messages (map/ground_truth)...")
        rospy.wait_for_message(self.topic_map, OccupancyGrid)
        rospy.wait_for_message(self.topic_ground_truth, Odometry)
        rospy.loginfo("[NavDataCollector] Ready. Will collect data to: %s" % self.data_root)



    # === 回调函数 ===
    def cmd_cb(self, msg):
        self.cmd = msg
        self._write_to_bag(self.topic_cmd, msg)

    def image_cb(self, msg):
        self.image = msg
        self._write_to_bag(self.topic_img, msg)

    def map_cb(self, msg):
        self.map_msg = msg

    def ground_truth_cb(self, msg):
        """接收来自 libgazebo_ros_p3d 的地面真实状态"""
        self.ground_truth = msg
        self._write_to_bag(self.topic_ground_truth, msg)

    def manual_goal_cb(self, msg):
        """接收手动下发的目标点 (MoveBaseActionGoal)。"""
        x = msg.goal.target_pose.pose.position.x
        y = msg.goal.target_pose.pose.position.y
        yaw = self.yaw_from_quat(msg.goal.target_pose.pose.orientation)

        self.manual_goal = (x, y, yaw)
        self.manual_goal_available = True
        rospy.loginfo("[NavDataCollector] Received manual goal: (%.2f, %.2f, %.2f)", x, y, yaw)

    def exec_goal_cb(self, msg):
        """监听move_base/goal上的正在执行的目标，并开始/结束记录episode。"""
        if self.goal_mode != "record_existing":
            return

        x = msg.goal.target_pose.pose.position.x
        y = msg.goal.target_pose.pose.position.y
        yaw = self.yaw_from_quat(msg.goal.target_pose.pose.orientation)
        new_goal = (x, y, yaw)

        # 如果是同一个目标，无需重复处理
        if self.current_goal is not None:
            # 检查是否是相同的目标 (位置误差很小)
            if np.hypot(self.current_goal[0] - x, self.current_goal[1] - y) < 0.1:
                return

        # 开启新的episode记录
        self.start_recording_episode(goal_pose=new_goal)
        self._write_to_bag(self.topic_goal, msg, getattr(msg.header, "stamp", rospy.Time.now()))
        rospy.loginfo("[NavDataCollector] Started recording episode with goal: (%.2f, %.2f, %.2f)",
                      x, y, yaw)

    def start_recording_episode(self, goal_pose=None):
        """开始记录一个新的episode"""
        with self.lock:
            if goal_pose is not None:
                self.current_goal = goal_pose

            if self.is_recording:
                rospy.logwarn("[NavDataCollector] Previous recording still active. Forcing finish before starting new episode.")
                self.finish_recording_episode(force=True)

            self.start_pose = self.get_robot_pose()
            self.is_recording = True
            self.last_episode_summary = None
            # 重置数据缓存
            self.episode_data = {
                "poses": [],
                "twists": [],
                "goal_rels": [],
                "dones": [],
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_name = f"episode_{self.episode_id:06d}_{timestamp}.bag"
            self.current_bag_path = os.path.join(self.data_root, bag_name)
            suffix = 1
            while os.path.exists(self.current_bag_path):
                bag_name = f"episode_{self.episode_id:06d}_{timestamp}_{suffix}.bag"
                self.current_bag_path = os.path.join(self.data_root, bag_name)
                suffix += 1

            try:
                self.current_bag = rosbag.Bag(self.current_bag_path, "w")
                rospy.loginfo("[NavDataCollector] Recording episode %d to %s", self.episode_id, self.current_bag_path)
            except Exception as exc:
                rospy.logerr("[NavDataCollector] Failed to create rosbag %s: %s", self.current_bag_path, str(exc))
                self.current_bag = None
                self.current_bag_path = None

            self.bag_message_count = 0

    def _write_to_bag(self, topic, msg, stamp=None):
        """将消息写入当前rosbag，如果正在记录"""
        with self.lock:
            if not self.is_recording or self.current_bag is None:
                return

            if stamp is None:
                msg_header = getattr(msg, "header", None)
                if msg_header is not None and getattr(msg_header, "stamp", rospy.Time()) != rospy.Time():
                    stamp = msg_header.stamp
                else:
                    stamp = rospy.Time.now()

            self.current_bag.write(topic, msg, stamp)
            self.bag_message_count += 1
    
    def step_record(self):
        """执行一步数据记录，如果正在记录episode"""
        with self.lock:
            if not self.is_recording or self.current_goal is None:
                return

            pose = self.get_robot_pose()
            twist = self.get_robot_twist()
            if pose is None or twist is None:
                return

            x, y, yaw = pose
            v, omega = twist

            goal_rel = self.compute_goal_rel(self.current_goal)
            if goal_rel is None:
                return

            # 检查是否到达目标
            dx = self.current_goal[0] - x
            dy = self.current_goal[1] - y
            dist_to_goal = math.hypot(dx, dy)
            done = 1 if dist_to_goal < 0.3 else 0

            # 记录数据
            self.episode_data["poses"].append([x, y, yaw])
            self.episode_data["twists"].append([v, omega])
            self.episode_data["goal_rels"].append(list(goal_rel))
            self.episode_data["dones"].append(done)

            # 如果到达目标，结束recording
            if done:
                self.finish_recording_episode()

    def finish_recording_episode(self, force=False):
        """结束记录当前episode并保存"""
        with self.lock:
            if not self.is_recording and not force:
                return

            success = False
            steps = 0
            if self.episode_data:
                success = any(d == 1 for d in self.episode_data["dones"])
                steps = len(self.episode_data["poses"])

            bag_path = self.current_bag_path
            if self.current_bag is not None:
                self.current_bag.close()
                self.current_bag = None

            if self.bag_message_count == 0 and bag_path and os.path.exists(bag_path):
                os.remove(bag_path)
                rospy.logwarn("[NavDataCollector] Removed empty rosbag %s", bag_path)
                bag_path = None
            
            if bag_path:
                rospy.loginfo("[NavDataCollector] Finished recording episode %d: success=%s steps=%d path=%s",
                            self.episode_id, success, steps, bag_path)
            else:
                rospy.loginfo("[NavDataCollector] Episode %d finished without saving rosbag (success=%s steps=%d)",
                            self.episode_id, success, steps)

            self.last_episode_summary = {
                "success": success,
                "steps": steps,
                "bag_path": bag_path,
            }

            self.episode_id += 1
            self.is_recording = False
            self.current_goal = None
            self.episode_data = None
            self.current_bag_path = None
            self.bag_message_count = 0
            self.start_pose = None

    # === 工具函数 ===
    @staticmethod
    def yaw_from_quat(q):
        import tf.transformations as tft
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = tft.euler_from_quaternion(quat)
        return yaw

    def get_robot_pose(self):
        """从 ground_truth/state (Odometry) 获取机器人位姿 (x, y, yaw)"""
        if self.ground_truth is None:
            rospy.logwarn("[NavDataCollector] ground_truth not available yet")
            return None
        
        # Odometry 消息中的位置和姿态
        p = self.ground_truth.pose.pose
        x = p.position.x
        y = p.position.y
        yaw = self.yaw_from_quat(p.orientation)
        return x, y, yaw

    def get_robot_twist(self):
        """返回 (v, omega)，从最近一次的 cmd_vel 消息获得"""
        if self.cmd is None:
            return 0.0, 0.0
        v = self.cmd.linear.x
        omega = self.cmd.angular.z
        return v, omega

    def compute_goal_rel(self, goal_pose):
        """goal_pose:(x_g,y_g,yaw_g) in map frame -> (x_rel,y_rel) in base frame"""
        robot = self.get_robot_pose()
        if robot is None:
            return None
        x_r, y_r, yaw_r = robot
        x_g, y_g, _ = goal_pose
        dx = x_g - x_r
        dy = y_g - y_r
        cos_y = math.cos(-yaw_r)
        sin_y = math.sin(-yaw_r)
        x_rel = cos_y * dx - sin_y * dy
        y_rel = sin_y * dx + cos_y * dy
        return x_rel, y_rel

    def sample_goal_from_map(self):
        """从 /map OccupancyGrid 中随机选一个 free cell 作为目标。"""
        if self.map_msg is None:
            rospy.logwarn("[NavDataCollector] No map yet.")
            return None
        grid = self.map_msg
        data = np.array(grid.data, dtype=np.int8)
        free_indices = np.where(data == 0)[0]
        if len(free_indices) == 0:
            rospy.logwarn("[NavDataCollector] No free cell in map.")
            return None
        idx = np.random.choice(free_indices)
        res = grid.info.resolution
        width = grid.info.width
        origin = grid.info.origin
        gx = idx % width
        gy = idx // width
        x = origin.position.x + (gx + 0.5) * res
        y = origin.position.y + (gy + 0.5) * res
        yaw = np.random.uniform(-math.pi, math.pi)
        return x, y, yaw

    def choose_goal_pose(self):
        """
        为当前 episode 选择目标点:
        - goal_mode == 'random': 始终随机采样
        - goal_mode == 'manual': 等待手动目标
        返回 (goal_pose:(x,y,yaw), is_manual:bool)
        """
        if self.goal_mode == "manual":
            if self.manual_goal_available and self.manual_goal is not None:
                goal_pose = self.manual_goal
                self.manual_goal_available = False
                rospy.loginfo("[NavDataCollector] Using manual goal for episode.")
                return goal_pose, True
            else:
                # 等待手动目标
                rospy.loginfo("[NavDataCollector] Waiting for manual goal...")
                rate = rospy.Rate(10.0)
                while not rospy.is_shutdown():
                    if self.manual_goal_available and self.manual_goal is not None:
                        goal_pose = self.manual_goal
                        self.manual_goal_available = False
                        rospy.loginfo("[NavDataCollector] Using manual goal for episode.")
                        return goal_pose, True
                    rate.sleep()
        elif self.goal_mode == "random":
            # 随机采样目标点
            goal_pose = self.sample_goal_from_map()
            return goal_pose, False
        else:
            # 默认随机采样
            goal_pose = self.sample_goal_from_map()
            return goal_pose, False

    def publish_goal(self, goal_pose):
        x, y, yaw = goal_pose
        from tf.transformations import quaternion_from_euler
        
        # 创建 MoveBaseActionGoal 消息
        action_goal = MoveBaseActionGoal()
        
        # Header 信息
        action_goal.header.frame_id = self.map_frame
        action_goal.header.stamp = rospy.Time.now()
        
        # Goal ID
        action_goal.goal_id.id = "goal_%d" % self.next_goal_id
        action_goal.goal_id.stamp = rospy.Time.now()
        self.next_goal_id += 1
        
        # Goal 位置和方向
        action_goal.goal.target_pose.header.frame_id = self.map_frame
        action_goal.goal.target_pose.header.stamp = rospy.Time.now()
        action_goal.goal.target_pose.pose.position.x = x
        action_goal.goal.target_pose.pose.position.y = y
        action_goal.goal.target_pose.pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, yaw)
        action_goal.goal.target_pose.pose.orientation.x = q[0]
        action_goal.goal.target_pose.pose.orientation.y = q[1]
        action_goal.goal.target_pose.pose.orientation.z = q[2]
        action_goal.goal.target_pose.pose.orientation.w = q[3]
        
        self.goal_pub.publish(action_goal)
        self._write_to_bag(self.topic_goal, action_goal, action_goal.header.stamp)
        rospy.loginfo("[NavDataCollector] Published goal: (%.2f, %.2f, %.2f)" % (x, y, yaw))

    def run_episode(self, episode_idx):
        """单个 episode 的数据采集。"""
        goal_pose, is_manual = self.choose_goal_pose()
        if goal_pose is None:
            rospy.logwarn("[NavDataCollector] Failed to get goal (mode=%s).", self.goal_mode)
            return False, 0

        start_pose = self.get_robot_pose()
        if start_pose is None:
            rospy.logwarn("[NavDataCollector] No robot pose, skip episode.")
            return False, 0

        rospy.loginfo("[NavDataCollector] Episode %d: start=(%.2f,%.2f) goal=(%.2f,%.2f) (manual=%s)",
                      episode_idx, start_pose[0], start_pose[1], goal_pose[0], goal_pose[1], str(is_manual))

        self.start_recording_episode(goal_pose=goal_pose)
        self.publish_goal(goal_pose)
        rospy.sleep(2.0)

        rate = rospy.Rate(self.record_rate)
        t_start = rospy.Time.now().to_sec()

        while not rospy.is_shutdown() and self.is_recording:
            now = rospy.Time.now().to_sec()
            if now - t_start > self.episode_timeout:
                rospy.loginfo("[NavDataCollector] Episode %d timeout.", episode_idx)
                break

            self.step_record()
            if not self.is_recording:
                break

            if self.episode_data and len(self.episode_data["poses"]) >= self.episode_max_steps:
                rospy.loginfo("[NavDataCollector] Episode %d reached max steps (%d).",
                              episode_idx, self.episode_max_steps)
                break

            rate.sleep()

        if self.is_recording:
            rospy.loginfo("[NavDataCollector] Episode %d forcing finish.", episode_idx)
            self.finish_recording_episode(force=True)

        summary = self.last_episode_summary or {}
        success = summary.get("success", False)
        steps = summary.get("steps", 0)
        return success, steps

    def run(self):
        """主循环：根据模式选择不同的运行方式"""
        if self.goal_mode == "record_existing":
            # record_existing模式：监听现有的goal，只负责记录
            rate = rospy.Rate(self.record_rate)
            while not rospy.is_shutdown():
                self.step_record()  # 尝试记录一步数据
                rate.sleep()
        else:
            # random模式：主动生成并执行episode
            episode_id = 0
            while not rospy.is_shutdown():
                success, steps = self.run_episode(episode_id)
                summary = self.last_episode_summary or {}
                bag_path = summary.get("bag_path")
                if bag_path:
                    rospy.loginfo("[NavDataCollector] Episode %d finished. success=%s steps=%d bag=%s",
                                  episode_id, success, steps, bag_path)
                else:
                    rospy.loginfo("[NavDataCollector] Episode %d finished. success=%s steps=%d",
                                  episode_id, success, steps)
                episode_id += 1
                rospy.sleep(2.0)


if __name__ == "__main__":
    collector = NavDataCollector()
    collector.run()
