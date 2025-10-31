#!/usr/bin/env python3
"""PPO environment manager built on navbot_ppo TD3 environment logic."""

import math
import os
import random

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Float32MultiArray
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel


DIAGONAL_DIS = math.sqrt(2.0) * (3.6 + 3.8)


class PPOTrainingEnvironment:
    """Gazebo-backed environment that mirrors navbot TD3 Env behavior for PPO."""

    def __init__(self):
        rospy.init_node('ppo_training_environment', anonymous=True)

        self.position = Pose().position
        self.goal_position = Pose()
        self.goal_position.position.x = 0.0
        self.goal_position.position.y = 0.0
        self.rel_theta = 0.0
        self.yaw = 0.0
        self.diff_angle = 0.0
        self.past_distance = 0.0
        self.past_actions = [0.0, 0.0]
        self.current_action = [0.0, 0.0]
        self.new_action_received = False
        self.episode_active = False
        self.start_pose = Pose()
        self.start_yaw = 0.0
        self.plan_waypoints = []

        self.threshold_arrive = 0.2
        self.min_scan_range = 0.2
        self.plan_point_count = rospy.get_param('~plan_point_count', 5)
        self.plan_feature_dim = self.plan_point_count * 2
        self.min_start_goal_distance = rospy.get_param('~min_start_goal_distance', 1.5)
        self.robot_model_name = rospy.get_param('~robot_model_name', 'turtlebot3_waffle')

        rospack = rospkg.RosPack()
        self.goal_model_xml = None

        model_path = rospy.get_param('~target_model_path', '')

        if not model_path:
            try:
                pkg_path = rospack.get_path('turtlebot3_gazebo')
                model_path = os.path.join(pkg_path, 'models', 'turtlebot3_waffle', 'model.sdf')
            except rospkg.ResourceNotFound:
                model_path = '/data/lzq/ros_motion_planning/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf'

        if os.path.exists(model_path):
            with open(model_path, 'r', encoding='utf-8') as model_file:
                self.goal_model_xml = model_file.read()
            rospy.loginfo(f'Loaded target model from {model_path}')
        else:
            rospy.logwarn(
                f'Target model.sdf not found at {model_path}. Set ~target_model_path if you need Gazebo visualization.'
            )

        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        self.state_pub = rospy.Publisher('/ppo/state', Float32MultiArray, queue_size=1)
        self.reward_pub = rospy.Publisher('/ppo/reward', Float32, queue_size=1)
        self.done_pub = rospy.Publisher('/ppo/done', Bool, queue_size=1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.cmd_sub = rospy.Subscriber('/ppo/cmd_vel', Twist, self.cmd_callback, queue_size=1)
        self.reset_sub = rospy.Subscriber('/ppo/env_reset', Bool, self.reset_callback, queue_size=1)

        rospy.loginfo('PPO Training Environment ready for commands')

    def reset_callback(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.handle_reset()

    def cmd_callback(self, msg: Twist) -> None:
        print("cmd callback triggered")
        self.current_action = [float(msg.linear.x), float(msg.angular.z)]
        self.new_action_received = True
        self.cmd_pub.publish(msg)

    def odom_callback(self, odom: Odometry) -> None:
        # print("odom callback triggered")
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z)))
        self.yaw = yaw if yaw >= 0 else yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        theta = math.atan2(rel_dis_y, rel_dis_x) #返回[-π, π] 
        self.rel_theta = round(math.degrees(theta), 2)
        diff_angle = abs(self.rel_theta - self.yaw)
        self.diff_angle = round(diff_angle if diff_angle <= 180 else 360 - diff_angle, 2)

    def scan_callback(self, scan: LaserScan) -> None:
        print("scan callback triggered")
        if not self.episode_active or not self.new_action_received:
            return

        state, rel_distance, done, arrived = self.get_state(scan)
        reward = self.compute_reward(done, arrived, rel_distance)
        done_flag = done or arrived

        self.publish_transition(state, reward, done_flag)
        self.past_actions = list(self.current_action)
        self.new_action_received = False

        if done_flag:
            self.episode_active = False
            self.stop_robot()
            if arrived:
                self.safe_delete_target()

    def handle_reset(self) -> None:
        rospy.loginfo('>>> handle_reset: START')
        rospy.loginfo('Reset request received')
        self.episode_active = False
        self.stop_robot()

        self.wait_for_service('/gazebo/pause_physics', self.pause_proxy)
        try:
            self.pause_proxy()
        except rospy.ServiceException as exc:
            rospy.logwarn(f'Gazebo pause failed: {exc}')

        self.wait_for_service('/gazebo/delete_model', self.delete_model)
        try:
            self.delete_model('target')
        except rospy.ServiceException:
            rospy.logwarn('Target deletion skipped (not found)')

        self.set_start_pose()
        self.spawn_target()

        self.wait_for_service('/move_base/clear_costmaps', self.clear_costmaps)
        try:
            self.clear_costmaps()
        except rospy.ServiceException as exc:
            rospy.logwarn(f'clear_costmaps failed: {exc}')

        self.plan_waypoints = []

        self.wait_for_service('/gazebo/unpause_physics', self.unpause_proxy)
        try:
            self.unpause_proxy()
        except rospy.ServiceException as exc:
            rospy.logerr(f'Gazebo unpause failed: {exc}')
            return

        scan = self.wait_for_scan()
        if scan is None:
            rospy.logerr('Failed to obtain initial scan during reset')
            return

        self.past_distance = self.get_goal_distance()
        self.past_actions = [0.0, 0.0]
        self.new_action_received = False
        self.episode_active = True

        self.compute_global_plan()

        state, _, _, _ = self.get_state(scan, include_done=False)
        self.publish_transition(state, 0.0, False)
        rospy.loginfo('>>> handle_reset: SUCCESS, initial state published.')
        rospy.loginfo('Environment reset complete')

    def wait_for_service(self, service_name: str, proxy) -> None:
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
        except rospy.ROSException:
            rospy.logerr(f'Service {service_name} unavailable')
            raise

    def wait_for_scan(self) -> LaserScan:
        try:
            return rospy.wait_for_message('/scan', LaserScan, timeout=5.0)
        except rospy.ROSException:
            return None

    def spawn_target(self) -> None:
        for _ in range(100):
            goal_x = random.uniform(-3.6, 3.6)
            goal_y = random.uniform(-3.6, 3.6)
            distance = math.hypot(goal_x - self.start_pose.position.x, goal_y - self.start_pose.position.y)
            if distance >= self.min_start_goal_distance:
                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
                self.goal_position.position.z = 0.0
                break
        else:
            rospy.logwarn('Using fallback goal position due to sampling failure')
            self.goal_position.position.x = self.start_pose.position.x + self.min_start_goal_distance
            self.goal_position.position.y = self.start_pose.position.y
            self.goal_position.position.z = 0.0

        if not self.goal_model_xml:
            rospy.logwarn_once('Skipping Gazebo target spawn because model XML is unavailable')
            return

        self.wait_for_service('/gazebo/spawn_sdf_model', self.spawn_model)
        try:
            self.spawn_model(
                model_name='target',
                model_xml=self.goal_model_xml,
                robot_namespace='ppo_target',
                initial_pose=self.goal_position,
                reference_frame='world',
            )
        except rospy.ServiceException as exc:
            rospy.logerr(f'Failed to spawn target: {exc}')

    def safe_delete_target(self) -> None:
        try:
            self.delete_model('target')
        except rospy.ServiceException:
            rospy.logwarn('Delete target failed during cleanup')

    def get_goal_distance(self) -> float:
        return math.hypot(
            self.goal_position.position.x - self.position.x,
            self.goal_position.position.y - self.position.y,
        )

    def get_state(self, scan: LaserScan, include_done: bool = True):
        scan_range = []
        for value in scan.ranges:
            if value == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(value):
                scan_range.append(0.0)
            else:
                scan_range.append(value)

        collision = self.min_scan_range > min(scan_range) > 0.0
        current_distance = self.get_goal_distance()
        arrived = current_distance <= self.threshold_arrive

        state = [reading / 3.5 for reading in scan_range]
        state.extend(self.past_actions)
        state.extend([
            current_distance / DIAGONAL_DIS,
            self.yaw / 360.0,
            self.rel_theta / 360.0,
            self.diff_angle / 180.0,
        ])
        state.extend(self.get_plan_features())

        done = collision
        if include_done:
            return np.asarray(state, dtype=np.float32), current_distance, done, arrived
        return np.asarray(state, dtype=np.float32), current_distance, False, arrived

    def compute_reward(self, collision: bool, arrived: bool, current_distance: float) -> float:
        distance_rate = self.past_distance - current_distance
        reward = 500.0 * distance_rate
        self.past_distance = current_distance

        if collision:
            reward = -100.0
        elif arrived:
            reward = 120.0
        return reward

    def publish_transition(self, state: np.ndarray, reward: float, done: bool) -> None:
        self.state_pub.publish(Float32MultiArray(data=state.tolist()))
        self.reward_pub.publish(Float32(data=float(reward)))
        self.done_pub.publish(Bool(data=done))

    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def set_start_pose(self) -> None:
        self.start_pose.position.x = random.uniform(-3.0, 3.0)
        self.start_pose.position.y = random.uniform(-3.0, 3.0)
        self.start_pose.position.z = 0.0
        self.start_yaw = random.uniform(-math.pi, math.pi)

        model_state = ModelState()
        model_state.model_name = self.robot_model_name
        model_state.pose.position.x = self.start_pose.position.x
        model_state.pose.position.y = self.start_pose.position.y
        model_state.pose.position.z = 0.0
        q_z = math.sin(self.start_yaw / 2.0)
        q_w = math.cos(self.start_yaw / 2.0)
        model_state.pose.orientation.z = q_z
        model_state.pose.orientation.w = q_w
        model_state.twist.linear.x = 0.0
        model_state.twist.linear.y = 0.0
        model_state.twist.linear.z = 0.0
        model_state.twist.angular.x = 0.0
        model_state.twist.angular.y = 0.0
        model_state.twist.angular.z = 0.0

        self.wait_for_service('/gazebo/set_model_state', self.set_model_state)
        try:
            self.set_model_state(model_state)
            self.position.x = self.start_pose.position.x
            self.position.y = self.start_pose.position.y
            self.position.z = 0.0
            self.yaw = math.degrees(self.start_yaw) % 360.0
        except rospy.ServiceException as exc:
            rospy.logerr(f'Set model state failed: {exc}')

    def compute_global_plan(self) -> None:
        rospy.loginfo('>>> compute_global_plan: START')
        self.plan_waypoints = []

        start = PoseStamped()
        start.header.frame_id = 'map'
        start.header.stamp = rospy.Time.now()
        start.pose.position.x = self.start_pose.position.x
        start.pose.position.y = self.start_pose.position.y
        q_z = math.sin(self.start_yaw / 2.0)
        q_w = math.cos(self.start_yaw / 2.0)
        start.pose.orientation.z = q_z
        start.pose.orientation.w = q_w

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = self.goal_position.position.x
        goal.pose.position.y = self.goal_position.position.y
        goal.pose.orientation.w = 1.0

        req = GetPlanRequest()
        req.start = start
        req.goal = goal
        req.tolerance = 0.1

        self.wait_for_service('/move_base/make_plan', self.make_plan)
        try:
            resp = self.make_plan(req)
        except rospy.ServiceException as exc:
            rospy.logwarn(f'>>> compute_global_plan: FAILED (ServiceException): {exc}')
            rospy.logwarn(f'make_plan failed: {exc}')
            return

        poses = resp.plan.poses
        if not poses:
            rospy.logwarn('>>> compute_global_plan: FAILED (Empty Plan)')
            rospy.logwarn('Received empty global plan')
            return

        if len(poses) <= self.plan_point_count:
            selected = poses
        else:
            indices = np.linspace(0, len(poses) - 1, self.plan_point_count, dtype=int)
            selected = [poses[i] for i in indices]

        self.plan_waypoints = [(pose.pose.position.x, pose.pose.position.y) for pose in selected]
        rospy.loginfo('>>> compute_global_plan: SUCCESS')

    def get_plan_features(self):
        features = []
        robot_x = self.position.x
        robot_y = self.position.y
        yaw_rad = math.radians(self.yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        for waypoint in self.plan_waypoints:
            dx = waypoint[0] - robot_x
            dy = waypoint[1] - robot_y
            rel_x = dx * cos_yaw + dy * sin_yaw
            rel_y = -dx * sin_yaw + dy * cos_yaw
            features.append(rel_x / DIAGONAL_DIS)
            features.append(rel_y / DIAGONAL_DIS)

        while len(features) < self.plan_feature_dim:
            features.append(0.0)

        return features


def main():
    env = PPOTrainingEnvironment()
    rospy.spin()


if __name__ == '__main__':
    main()