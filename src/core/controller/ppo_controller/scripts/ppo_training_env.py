#!/usr/bin/env python3
"""
PPO Training Environment Manager
Manages the simulation environment for PPO training, based on navbot_ppo best practices
"""

import rospy
import roslaunch
import rosparam
import time
import random
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Point, Pose
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import OccupancyGrid
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Bool

class PPOTrainingEnvironment:
    """
    管理PPO训练环境的类，参考navbot_ppo实现
    负责重置环境、设置目标点、检测碰撞等
    """
    
    def __init__(self):
        rospy.init_node('ppo_training_environment', anonymous=True)
        
        # Gazebo服务客户端 - 参考navbot_ppo的实现
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.pause_physics_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # 发布器
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.episode_reset_pub = rospy.Publisher('/ppo/reset', Bool, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # 订阅器
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 环境状态
        self.map_data = None
        self.free_spaces = []
        self.training_areas = self.define_training_areas()
        self.goal_position = Pose()
        self.robot_position = Pose()
        
        # 参考navbot_ppo的训练配置
        self.threshold_arrive = 0.2  # 到达目标的阈值
        self.collision_threshold = 0.2  # 碰撞检测阈值
        self.max_episode_steps = 500  # 最大步数
        self.current_episode_steps = 0
        
        # 环境边界 - 仓库环境
        self.env_bounds = {
            'x_min': -4.0, 'x_max': 4.0,
            'y_min': -4.0, 'y_max': 4.0
        }
        
        # 已知障碍物区域 - 避免在这些区域生成目标
        self.obstacle_areas = [
            {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0},  # 中心区域
            # 可以根据实际仓库布局添加更多障碍物区域
        ]
        
        rospy.loginfo("PPO Training Environment initialized")
    
    def define_training_areas(self):
        """定义训练区域"""
        # 可以根据不同地图定义不同的训练区域
        # 这里定义warehouse地图的一些训练区域
        return [
            {"start_area": (-8, -6, 8, 6), "goal_area": (-8, -6, 8, 6)},  # 整个地图
            {"start_area": (-5, -3, 0, 3), "goal_area": (0, -3, 5, 3)},   # 左右两侧
            {"start_area": (-3, -5, 3, 0), "goal_area": (-3, 0, 3, 5)},   # 上下两侧
        ]
        
        rospy.loginfo("PPO Training Environment Manager initialized")
        
    def map_callback(self, msg):
        """地图回调函数"""
        self.map_data = msg
        self.extract_free_spaces()
        
    def extract_free_spaces(self):
        """提取地图中的自由空间"""
        if self.map_data is None:
            return
            
        self.free_spaces = []
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        for i in range(0, len(self.map_data.data), 10):  # 采样以提高效率
            if self.map_data.data[i] == 0:  # 自由空间
                x_idx = i % width
                y_idx = i // width
                x = origin_x + x_idx * resolution
                y = origin_y + y_idx * resolution
                
                # 确保周围也是自由空间
                if self.is_safe_position(x, y, 0.5):
                    self.free_spaces.append((x, y))
        
        rospy.loginfo(f"Found {len(self.free_spaces)} free spaces")
    
    def is_safe_position(self, x, y, safety_radius=0.5):
        """检查位置是否安全（周围没有障碍物）"""
        if self.map_data is None:
            return False
            
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        # 转换为地图坐标
        map_x = int((x - origin_x) / resolution)
        map_y = int((y - origin_y) / resolution)
        
        # 检查安全半径内的所有点
        safety_cells = int(safety_radius / resolution)
        for dx in range(-safety_cells, safety_cells + 1):
            for dy in range(-safety_cells, safety_cells + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                if (check_x < 0 or check_x >= width or 
                    check_y < 0 or check_y >= height):
                    return False
                    
                idx = check_y * width + check_x
                if idx >= len(self.map_data.data) or self.map_data.data[idx] != 0:
                    return False
        
        return True
    
    def is_position_in_obstacle_area(self, x, y):
        """检查位置是否在障碍物区域内"""
        for area in self.obstacle_areas:
            if (area['x_min'] <= x <= area['x_max'] and 
                area['y_min'] <= y <= area['y_max']):
                return True
        return False
    
    def generate_safe_goal_position(self):
        """生成安全的目标位置，参考navbot_ppo实现"""
        max_attempts = 100
        for _ in range(max_attempts):
            # 在环境边界内生成随机位置
            x = random.uniform(self.env_bounds['x_min'], self.env_bounds['x_max'])
            y = random.uniform(self.env_bounds['y_min'], self.env_bounds['y_max'])
            
            # 检查是否在障碍物区域内
            if self.is_position_in_obstacle_area(x, y):
                continue
                
            # 检查是否是安全位置
            if self.is_safe_position(x, y, 0.3):
                return x, y
        
        # 如果找不到安全位置，返回默认位置
        rospy.logwarn("Could not find safe goal position, using default")
        return 2.0, 2.0
    
    def generate_safe_start_position(self):
        """生成安全的起始位置"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(self.env_bounds['x_min'], self.env_bounds['x_max'])
            y = random.uniform(self.env_bounds['y_min'], self.env_bounds['y_max'])
            
            # 确保起始位置不在障碍物区域内
            if self.is_position_in_obstacle_area(x, y):
                continue
                
            # 确保起始位置与目标位置有足够距离
            goal_distance = math.hypot(
                x - self.goal_position.position.x,
                y - self.goal_position.position.y
            )
            if goal_distance < 1.0:  # 最小距离1米
                continue
                
            if self.is_safe_position(x, y, 0.3):
                return x, y
        
        # 默认起始位置
        rospy.logwarn("Could not find safe start position, using default")
        return -2.0, -2.0
    
    def reset_environment(self):
        """重置环境 - 参考navbot_ppo的完整重置流程"""
        try:
            rospy.loginfo("Starting environment reset...")
            
            # 步骤1: 停止机器人
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            
            # 步骤2: 暂停物理仿真
            rospy.wait_for_service('/gazebo/pause_physics', timeout=5.0)
            self.pause_physics_service()
            
            # 步骤3: 重置Gazebo仿真
            rospy.wait_for_service('/gazebo/reset_simulation', timeout=5.0)
            self.reset_simulation_service()
            rospy.loginfo("Gazebo simulation reset")
            
            # 等待重置完成
            time.sleep(0.5)
            
            # 步骤4: 生成新的目标位置
            goal_x, goal_y = self.generate_safe_goal_position()
            self.goal_position.position.x = goal_x
            self.goal_position.position.y = goal_y
            self.goal_position.position.z = 0.0
            
            # 步骤5: 生成新的起始位置
            start_x, start_y = self.generate_safe_start_position()
            start_yaw = random.uniform(0, 2 * math.pi)  # 随机朝向
            
            # 步骤6: 恢复物理仿真
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=5.0)
            self.unpause_physics_service()
            
            # 等待物理仿真稳定
            time.sleep(0.2)
            
            # 步骤7: 重置机器人位置
            success = self.reset_robot_position(start_x, start_y, start_yaw)
            if not success:
                rospy.logwarn("Failed to reset robot position")
                return False
            
            # 步骤8: 设置新目标
            self.set_goal(goal_x, goal_y)
            
            # 步骤9: 发布重置信号给PPO agent
            self.episode_reset_pub.publish(Bool(data=True))
            
            # 重置计数器
            self.current_episode_steps = 0
            
            rospy.loginfo(f"Environment reset complete - Start: ({start_x:.2f}, {start_y:.2f}), Goal: ({goal_x:.2f}, {goal_y:.2f})")
            return True
            
        except Exception as e:
            rospy.logerr(f"Environment reset failed: {e}")
            return False
    
    def reset_to_training_area(self, area_idx=None):
        """重置到指定的训练区域"""
        if area_idx is None:
            area_idx = random.randint(0, len(self.training_areas) - 1)
        
        if area_idx >= len(self.training_areas):
            rospy.logwarn(f"Invalid area index {area_idx}, using random area")
            area_idx = 0
        
        training_area = self.training_areas[area_idx]
        
        try:
            # 暂停物理仿真
            rospy.wait_for_service('/gazebo/pause_physics', timeout=5.0)
            self.pause_physics_service()
            
            # 重置仿真
            rospy.wait_for_service('/gazebo/reset_simulation', timeout=5.0)
            self.reset_simulation_service()
            
            time.sleep(0.5)
            
            # 在指定区域内生成起始和目标位置
            start_x, start_y = self.get_random_position_in_area(training_area["start_area"])
            goal_x, goal_y = self.get_random_position_in_area(training_area["goal_area"])
            
            # 确保起始位置和目标位置有足够距离
            while math.hypot(goal_x - start_x, goal_y - start_y) < 1.0:
                goal_x, goal_y = self.get_random_position_in_area(training_area["goal_area"])
            
            start_yaw = random.uniform(0, 2 * math.pi)
            
            # 恢复物理仿真
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=5.0)
            self.unpause_physics_service()
            
            time.sleep(0.2)
            
            # 重置机器人和目标
            self.reset_robot_position(start_x, start_y, start_yaw)
            self.set_goal(goal_x, goal_y)
            
            # 更新目标位置
            self.goal_position.position.x = goal_x
            self.goal_position.position.y = goal_y
            
            # 发布重置信号
            self.episode_reset_pub.publish(Bool(data=True))
            
            # 重置计数器
            self.current_episode_steps = 0
            
            rospy.loginfo(f"Reset to training area {area_idx} - Start: ({start_x:.2f}, {start_y:.2f}), Goal: ({goal_x:.2f}, {goal_y:.2f})")
            return True
            
        except Exception as e:
            rospy.logerr(f"Training area reset failed: {e}")
            return False
    
    def get_random_position_in_area(self, area):
        """在指定区域内获取随机位置"""
        x_min, y_min, x_max, y_max = area
        
        # 从自由空间中筛选符合区域要求的位置
        valid_positions = [
            (x, y) for x, y in self.free_spaces
            if x_min <= x <= x_max and y_min <= y <= y_max
        ]
        
        if not valid_positions:
            # 如果没有找到，则在区域内随机生成
            for _ in range(100):  # 最多尝试100次
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                if self.is_safe_position(x, y):
                    return x, y
            
            # 如果还是找不到，返回区域中心
            return (x_min + x_max) / 2, (y_min + y_max) / 2
        
        return random.choice(valid_positions)
    
    def reset_robot_position(self, x, y, yaw=0.0):
        """重置机器人位置"""
        try:
            # 设置模型状态
            model_state = ModelState()
            model_state.model_name = "turtlebot3_waffle"
            model_state.pose.position.x = x
            model_state.pose.position.y = y
            model_state.pose.position.z = 0.0
            model_state.pose.orientation.z = np.sin(yaw / 2)
            model_state.pose.orientation.w = np.cos(yaw / 2)
            
            # 重置速度
            model_state.twist.linear.x = 0.0
            model_state.twist.linear.y = 0.0
            model_state.twist.angular.z = 0.0
            
            self.set_model_state_service(model_state)
            
            # 发布初始位姿（用于AMCL）
            initial_pose = PoseWithCovarianceStamped()
            initial_pose.header.stamp = rospy.Time.now()
            initial_pose.header.frame_id = "map"
            initial_pose.pose.pose.position.x = x
            initial_pose.pose.pose.position.y = y
            initial_pose.pose.pose.orientation.z = np.sin(yaw / 2)
            initial_pose.pose.pose.orientation.w = np.cos(yaw / 2)
            
            # # 设置协方差
            # initial_pose.pose.covariance[0] = 0.25  # x
            # initial_pose.pose.covariance[7] = 0.25  # y
            # initial_pose.pose.covariance[35] = 0.068  # yaw
            
            self.initial_pose_pub.publish(initial_pose)
            
            rospy.loginfo(f"Robot reset to position: ({x:.2f}, {y:.2f}, {yaw:.2f})")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to reset robot position: {e}")
            return False
    
    def set_goal(self, x, y, yaw=0.0):
        """设置目标点"""
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = np.sin(yaw / 2)
        goal.pose.orientation.w = np.cos(yaw / 2)
        
        self.goal_pub.publish(goal)
        rospy.loginfo(f"Goal set to: ({x:.2f}, {y:.2f}, {yaw:.2f})")
    
    def get_goal_distance(self):
        """获取机器人到目标的距离"""
        try:
            # 获取机器人当前位置
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rospy.Time(), rospy.Duration(1.0)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # 计算距离
            distance = math.hypot(
                self.goal_position.position.x - robot_x,
                self.goal_position.position.y - robot_y
            )
            return distance
            
        except Exception as e:
            rospy.logwarn(f"Failed to get goal distance: {e}")
            return float('inf')
    
    def check_goal_reached(self):
        """检查是否到达目标"""
        distance = self.get_goal_distance()
        return distance < self.threshold_arrive
    
    def check_collision(self):
        """检查是否发生碰撞"""
        # 这里可以添加更复杂的碰撞检测逻辑
        # 例如检查激光雷达数据、地图数据等
        try:
            # 获取机器人当前位置
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rospy.Time(), rospy.Duration(1.0)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # 检查是否在安全位置
            return not self.is_safe_position(robot_x, robot_y, self.collision_threshold)
            
        except Exception as e:
            rospy.logwarn(f"Failed to check collision: {e}")
            return False
    
    def check_episode_timeout(self):
        """检查回合是否超时"""
        return self.current_episode_steps >= self.max_episode_steps
    
    def step_episode(self):
        """推进回合一步"""
        self.current_episode_steps += 1
        
        # 检查回合结束条件
        goal_reached = self.check_goal_reached()
        collision = self.check_collision()
        timeout = self.check_episode_timeout()
        
        done = goal_reached or collision or timeout
        
        # 计算奖励
        reward = 0.0
        if goal_reached:
            reward = 100.0
            rospy.loginfo("Goal reached!")
        elif collision:
            reward = -100.0
            rospy.logwarn("Collision detected!")
        elif timeout:
            reward = -10.0
            rospy.logwarn("Episode timeout!")
        else:
            # 基于距离的奖励
            distance = self.get_goal_distance()
            reward = -distance * 0.1  # 距离越近奖励越高
        
        return reward, done, {
            'goal_reached': goal_reached,
            'collision': collision,
            'timeout': timeout,
            'distance': self.get_goal_distance(),
            'steps': self.current_episode_steps
        }
    
    def reset_episode(self, training_area_idx=None):
        """重置训练回合"""
        try:
            # 选择训练区域
            if training_area_idx is None:
                training_area_idx = random.randint(0, len(self.training_areas) - 1)
            
            training_area = self.training_areas[training_area_idx]
            
            # 获取起始和目标位置
            start_x, start_y = self.get_random_position_in_area(training_area["start_area"])
            goal_x, goal_y = self.get_random_position_in_area(training_area["goal_area"])
            
            # 确保起始和目标位置有足够距离
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance < 2.0:  # 最小距离2米
                # 重新选择目标点
                for _ in range(10):
                    goal_x, goal_y = self.get_random_position_in_area(training_area["goal_area"])
                    distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                    if distance >= 2.0:
                        break
            
            # 随机起始朝向
            start_yaw = random.uniform(-np.pi, np.pi)
            
            # 重置机器人位置
            if self.reset_robot_position(start_x, start_y, start_yaw):
                time.sleep(1.0)  # 等待位置稳定
                
                # 设置目标点
                self.set_goal(goal_x, goal_y)
                
                # 更新目标位置记录
                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
                
                # 发布重置信号给PPO代理
                self.episode_reset_pub.publish(Bool(data=True))
                
                # 重置计数器
                self.current_episode_steps = 0
                
                rospy.loginfo(f"Episode reset: Start({start_x:.2f}, {start_y:.2f}) -> Goal({goal_x:.2f}, {goal_y:.2f})")
                return True
            
            return False
            
        except Exception as e:
            rospy.logerr(f"Failed to reset episode: {e}")
            return False
    
    def start_training_session(self, max_episodes=1000):
        """开始训练会话"""
        rospy.loginfo(f"Starting PPO training session with {max_episodes} episodes")
        
        episode_count = 0
        
        try:
            while not rospy.is_shutdown() and episode_count < max_episodes:
                rospy.loginfo(f"Starting episode {episode_count + 1}/{max_episodes}")
                
                # 重置环境
                if self.reset_episode():
                    episode_count += 1
                    
                    # 等待回合完成（这里可以添加更复杂的逻辑）
                    time.sleep(30.0)  # 每回合最多30秒
                else:
                    rospy.logwarn("Failed to reset episode, retrying...")
                    time.sleep(1.0)
                    
        except KeyboardInterrupt:
            rospy.loginfo("Training interrupted by user")
        
        rospy.loginfo("Training session completed")

def main():
    try:
        env_manager = PPOTrainingEnvironment()
        
        # 等待地图数据
        rospy.loginfo("Waiting for map data...")
        while env_manager.map_data is None and not rospy.is_shutdown():
            time.sleep(0.1)
        
        rospy.loginfo("Map data received, starting training environment")
        
        # 开始训练
        env_manager.start_training_session(max_episodes=100)
        
    except Exception as e:
        rospy.logerr(f"Training environment error: {e}")

if __name__ == "__main__":
    main()