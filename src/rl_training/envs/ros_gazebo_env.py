import torch
import numpy as np
import rospy
import tf.transformations
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelState
import threading
import math

class RosGazeboEnv(VecEnv):
    def __init__(self, env_cfg, device="cpu"):
        self.env_cfg = env_cfg
        self.device = device
        
        # Extract robot configurations
        self.agent_names = env_cfg.get('agent_names')
        self.num_envs = len(self.agent_names)
        
        # Observation and Action spaces
        self.num_obs = env_cfg.get('num_observations')
        self.num_actions = env_cfg.get('num_actions')
        self.max_episode_length = env_cfg.get('max_episode_length')
        
        # ROS Initialization
        if not rospy.get_node_uri():
            rospy.init_node("rl_training_node", anonymous=True)
            
        # Gazebo Services
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_properties_srv = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        
        # Initialize Robots
        self.robots = []
        init_poses = env_cfg.get('init_poses', {})
        for i, name in enumerate(self.agent_names):
            init_pos = init_poses.get(name, [0.0, 0.0, 0.0])
            self.robots.append(RobotAgent(name, i, name, env_cfg, init_pos))
            
        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.rew_buf = torch.zeros(self.num_envs, device=device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.extras = {}
        
    def get_observations(self):
        for i, robot in enumerate(self.robots):
            obs = robot.get_observation()
            self.obs_buf[i] = torch.tensor(obs, device=self.device)
            
        return TensorDict({
            "policy": self.obs_buf,
            "privileged": self.obs_buf.clone() 
        }, batch_size=[self.num_envs])

    def step(self, actions):
        actions_np = actions.detach().cpu().numpy()
        
        for i, robot in enumerate(self.robots):
            robot.set_action(actions_np[i])
            
        rospy.sleep(0.01) 
        
        obs = self.get_observations()
        
        self.rew_buf[:] = 0.0
        self.reset_buf[:] = False
        
        for i, robot in enumerate(self.robots):
            rew, done = robot.compute_reward_and_done()
            self.rew_buf[i] = rew
            self.reset_buf[i] = done
            
            if done:
                self.reset_robot(i)
                self.episode_length_buf[i] = 0
            else:
                self.episode_length_buf[i] += 1
                
        time_outs = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= time_outs
        
        for i in range(self.num_envs):
            if time_outs[i]:
                self.reset_robot(i)
                self.episode_length_buf[i] = 0

        return obs, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        for i in range(self.num_envs):
            self.reset_robot(i)
        self.episode_length_buf[:] = 0
        self.reset_buf[:] = False
        return self.get_observations()

    def reset_robot(self, idx):
        robot = self.robots[idx]
        init_x, init_y, _ = robot.init_pos
        
        # Reset Pose
        state = ModelState()
        state.model_name = robot.model_name 
        state.pose.position.x = init_x
        state.pose.position.y = init_y
        state.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, np.random.uniform(-3.14, 3.14))
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]
        
        try:
            self.set_model_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            
        # Generate New Goal
        min_dist = self.env_cfg.get('goal_min_dist')
        max_dist = self.env_cfg.get('goal_max_dist')
        
        goal_dist = np.random.uniform(min_dist, max_dist)
        goal_angle = np.random.uniform(-3.14, 3.14)
        goal_x = state.pose.position.x + goal_dist * math.cos(goal_angle)
        goal_y = state.pose.position.y + goal_dist * math.sin(goal_angle)
        
        robot.reset(goal_x, goal_y, init_x, init_y)

class RobotAgent:
    def __init__(self, namespace, id, model_name, env_cfg, init_pos):
        self.ns = namespace
        self.id = id
        self.model_name = model_name
        self.init_pos = init_pos
        self.env_cfg = env_cfg
        self.reward_cfg = env_cfg.get('reward')
        self.term_cfg = env_cfg.get('termination')
        
        self.cmd_vel_pub = rospy.Publisher(f"/{self.ns}/cmd_vel", Twist, queue_size=1)
        self.goal_marker_pub = rospy.Publisher(f"/{self.ns}/rl_goal_marker", Marker, queue_size=1)
        
        self.scan = np.zeros(360)
        self.odom = None
        
        rospy.Subscriber(f"/{self.ns}/scan", LaserScan, self.scan_cb)
        rospy.Subscriber(f"/{self.ns}/odom", Odometry, self.odom_cb)
        
        self.lock = threading.Lock()
        
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.prev_dist_to_goal = 0.0
        self.prev_pose = np.zeros(2)
        
        self.current_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.success_steps = 0

    def scan_cb(self, msg):
        with self.lock:
            scan = np.array(msg.ranges)
            if len(scan) > 360:
                scan = scan[:360]
            elif len(scan) < 360:
                scan = np.pad(scan, (0, 360-len(scan)), 'constant')
            self.scan = np.nan_to_num(scan, posinf=100.0, neginf=0.0)

    def odom_cb(self, msg):
        with self.lock:
            self.odom = msg

    def get_observation(self):
        with self.lock:
            target_dist = 0.0
            target_angle = 0.0
            lin_vel = 0.0
            ang_vel = 0.0
            
            if self.odom is not None:
                px = self.odom.pose.pose.position.x
                py = self.odom.pose.pose.position.y
                yaw = self.get_yaw(self.odom.pose.pose.orientation)
                
                lin_vel = self.odom.twist.twist.linear.x
                ang_vel = self.odom.twist.twist.angular.z
                
                dx = self.goal_x - px
                dy = self.goal_y - py
                target_dist = math.sqrt(dx**2 + dy**2)
                target_angle = math.atan2(dy, dx) - yaw
                target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))

            obs = np.concatenate([
                self.scan, 
                [target_dist, target_angle, lin_vel, ang_vel]
            ]) 
            return obs

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def set_action(self, action):
        self.prev_action = self.current_action
        self.current_action = action

        clip_low = self.env_cfg.get('action_clip_low')
        clip_high = self.env_cfg.get('action_clip_high')
        
        msg = Twist()
        msg.linear.x = np.clip(action[0], clip_low[0], clip_high[0]) 
        msg.angular.z = np.clip(action[1], clip_low[1], clip_high[1])
        
        if self.env_cfg.get('reward_debug', False):
             print(f"Robot {self.id} Raw Action: [{action[0]:.2f}, {action[1]:.2f}] -> Clipped: [{msg.linear.x:.2f}, {msg.angular.z:.2f}]")

        self.cmd_vel_pub.publish(msg)



    def compute_reward_and_done(self):
        reward = 0.0
        done = False

        # Configs
        r_cfg = self.reward_cfg
        t_cfg = self.term_cfg

        # State
        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        pz = self.odom.pose.pose.position.z
        yaw = self.get_yaw(self.odom.pose.pose.orientation)
        lin_vel = self.odom.twist.twist.linear.x
        ang_vel = self.odom.twist.twist.angular.z
        
        dist_to_goal = math.sqrt((self.goal_x - px)**2 + (self.goal_y - py)**2)
        
        # Target Angle
        dx = self.goal_x - px
        dy = self.goal_y - py
        target_angle = math.atan2(dy, dx) - yaw
        target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle)) # Normalize

        # --- Reward Calculation ---
        
        # 0. Projected Displacement Reward
        distance_scale = r_cfg.get('distance_scale')
        displacement_reward = 0.0
        if np.any(self.prev_pose):
            dx_move = px - self.prev_pose[0]
            dy_move = py - self.prev_pose[1]
            
            dx_target = self.goal_x - self.prev_pose[0]
            dy_target = self.goal_y - self.prev_pose[1]
            dist_to_target = math.sqrt(dx_target**2 + dy_target**2) + 1e-8
            
            unit_x = dx_target / dist_to_target
            unit_y = dy_target / dist_to_target
            
            displacement_projection = dx_move * unit_x + dy_move * unit_y
            displacement_reward = displacement_projection * distance_scale
            reward += displacement_reward

        # 1. Displacement / Progress Reward
        dist_reward_scale = r_cfg.get('dist_reward_scale')
        progress = self.prev_dist_to_goal - dist_to_goal
        progress_reward = progress * dist_reward_scale
        reward += progress_reward
        
        # 2. Step Cost
        step_cost = r_cfg.get('step_cost')
        reward += step_cost
        
        # 3. Smoothness
        # smoothness_scale is negative, penalizing difference
        diff = self.current_action - self.prev_action
        smoothness_reward = np.dot(diff, diff) * r_cfg.get('smoothness_scale')
        reward += smoothness_reward
        
        # 4. Heading Penalty
        heading_penalty = abs(target_angle) * r_cfg.get('heading_penalty_scale')
        reward += heading_penalty
        
        # 5. Collision
        min_scan = np.min(self.scan)
        collision_dist = t_cfg.get('min_collision_range')
        collision_reward = 0.0
        
        if min_scan < collision_dist:
            collision_reward = r_cfg.get('collision_penalty')
            reward += collision_reward
            if t_cfg.get('collision'):
                done = True
        
        # 6. Min Range Penalty (Obstacle avoidance)
        min_range_threshold = r_cfg.get('min_range_threshold')
        min_range_reward = 0.0
        if min_scan < min_range_threshold:
             shortfall = min_range_threshold - min_scan
             ratio = shortfall / max(min_range_threshold, 1e-6)
             min_range_reward = r_cfg.get('min_range_penalty') * ratio
             reward += min_range_reward

        # 7. Goal & Success
        # Success Check
        at_goal_pos = dist_to_goal < t_cfg.get('success_pos')
        at_goal_yaw = abs(target_angle) < t_cfg.get('success_yaw')
        stopped = abs(lin_vel) < t_cfg.get('success_lin_vel_th') and abs(ang_vel) < t_cfg.get('success_ang_vel_th')
        
        goal_reward = r_cfg.get('goal_reward')
        current_goal_reward = 0.0
        
        # Position and orientation both achieved
        if at_goal_pos and at_goal_yaw and stopped:
            current_goal_reward += goal_reward
            self.success_steps += 1
        else:
            self.success_steps = 0
            
        # Only achieved position
        if at_goal_pos:
            current_goal_reward += goal_reward * 0.5

        reward += current_goal_reward

        if self.success_steps >= t_cfg.get('success_stay_steps'):
            done = True
            
        # Pose-based reward within a gate radius
        pose_reward = 0.0
        if dist_to_goal < r_cfg.get('pose_gate_radius'):
            yaw_reward_scale = r_cfg.get('yaw_reward_scale')
            pose_reward = yaw_reward_scale * (np.exp(abs(target_angle)) - 1.0)
            reward += pose_reward
            
        # Termination: Min Height
        if pz < t_cfg.get('min_height'):
            done = True
            
        # Update prev
        self.prev_dist_to_goal = dist_to_goal
        self.prev_pose = np.array([px, py])
        
        if self.env_cfg.get('reward_debug', False):
            print("--- Reward Debug ---")
            print(f"Robot {self.id} Reward: {reward:.4f} | "
                  f"Disp: {displacement_reward:.4f} | "
                  f"Prog: {progress_reward:.4f} | "
                  f"Step: {step_cost:.4f} | "
                  f"Smooth: {smoothness_reward:.4f} | "
                  f"Head: {heading_penalty:.4f} | "
                  f"Coll: {collision_reward:.4f} | "
                  f"MinReward: {min_range_reward:.4f} | "
                  f"Goal: {current_goal_reward:.4f} | "
                  f"Pose: {pose_reward:.4f} | "
                  f"Done: {done}") 
        
        return reward, done

    def reset(self, goal_x, goal_y, current_x, current_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        
        # Reset state variables
        self.prev_pose = np.array([current_x, current_y])
        self.prev_dist_to_goal = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        
        self.current_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.success_steps = 0
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rl_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal_x
        marker.pose.position.y = goal_y
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.goal_marker_pub.publish(marker)
