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
from std_srvs.srv import Empty
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
        self.control_dt = env_cfg.get('control_dt')

        # ROS Initialization
        if not rospy.get_node_uri():
            rospy.init_node("rl_training_node", anonymous=True)
            
        # Gazebo Services
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')

        self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_properties_srv = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        self.pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        
        # Initialize Robots
        self.robots = []
        self.init_poses = env_cfg.get('init_poses', {})
        for i, name in enumerate(self.agent_names):
            init_pos = self.init_poses.get(name, [0.0, 0.0, 0.0])
            self.robots.append(RobotAgent(name, i, name, env_cfg, init_pos))

        # Wait for odom to ensure valid observations at startup
        rospy.loginfo("Waiting for robots to receive odometry...")
        for robot in self.robots:
            robot.wait_for_odom()
        rospy.loginfo("All robots ready.")

        # Track robot1 init pose for interference goal reference
        self.robot1_init = self.init_poses.get(self.agent_names[0], [0.0, 0.0, 0.0]) if self.agent_names else [0.0, 0.0, 0.0]

        # Optional interference robot (e.g., robot2) control
        opponent_cfg = env_cfg.get('opponent', {})
        self.opponent_enabled = opponent_cfg.get('enabled', False)
        self.opponent_name = opponent_cfg.get('model_name', 'robot2')
        self.opponent_goal_topic = opponent_cfg.get('goal_topic', f"/{self.opponent_name}/move_base_simple/goal")
        self.opponent_frame_id = opponent_cfg.get('frame_id', 'map')
        self.opponent_goal_offset = opponent_cfg.get('goal_offset', 0.3)
        self.opponent_init_pose = self.init_poses.get(self.opponent_name, opponent_cfg.get('init_pose', [0.0, 0.0, 0.0]))
        self.opponent_goal_pub = None
        if self.opponent_enabled:
            self.opponent_goal_pub = rospy.Publisher(self.opponent_goal_topic, PoseStamped, queue_size=1)
            
        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
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
        
        # 1. Set actions (publish cmd_vel)
        for i, robot in enumerate(self.robots):
            robot.set_action(actions_np[i])
            
        # 2. Unpause physics to execute actions
        try:
            self.unpause_physics_srv()
        except rospy.ServiceException as e:
            rospy.logerr(f"/gazebo/unpause_physics service call failed: {e}")

        # 3. Wait for control_dt
        rospy.sleep(self.control_dt)
        
        # 4. Pause physics to freeze state
        try:
            self.pause_physics_srv()
        except rospy.ServiceException as e:
            rospy.logerr(f"/gazebo/pause_physics service call failed: {e}")

        # 5. Get observations
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
        # Unpause to allow state updates during reset
        try:
            self.unpause_physics_srv()
        except rospy.ServiceException as e:
            rospy.logerr(f"/gazebo/unpause_physics service call failed: {e}")

        for i in range(self.num_envs):
            self.reset_robot(i)
        
        # Wait for sensors to update after reset
        rospy.sleep(self.control_dt)

        # Pause again
        try:
            self.pause_physics_srv()
        except rospy.ServiceException as e:
            rospy.logerr(f"/gazebo/pause_physics service call failed: {e}")

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
        
        # Reset Velocity
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        try:
            self.set_model_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            
        # Generate New Goal
        goals = self.env_cfg.get('goals')

        # 从配置的列表中随机选择一个目标点
        goal_idx = np.random.randint(0, len(goals))
        goal = goals[goal_idx]
        goal_x = goal[0]
        goal_y = goal[1]
        goal_yaw = goal[2]
        robot.reset(goal_x, goal_y, goal_yaw, init_x, init_y)

        # Also reset and command the interference robot (e.g., robot2)
        if self.opponent_enabled and robot.model_name == self.agent_names[0]:
            self.reset_interference_robot()

    def reset_interference_robot(self):
        if not self.opponent_enabled:
            return

        # Reset opponent pose in Gazebo
        state = ModelState()
        state.model_name = self.opponent_name
        state.pose.position.x = self.opponent_init_pose[0]
        state.pose.position.y = self.opponent_init_pose[1]
        state.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, self.opponent_init_pose[2])
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]

        try:
            self.set_model_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logerr(f"[reset_interference_robot] Service call failed: {e}")

        # Send goal for opponent: robot1 initial pose as its destination
        goal_x = self.robot1_init[0]
        goal_y = self.robot1_init[1]
        self.publish_interference_goal(goal_x, goal_y)

    def publish_interference_goal(self, goal_x, goal_y):
        """Send goal to opponent with a small offset to avoid full footprint overlap at the target."""
        if not self.opponent_enabled or self.opponent_goal_pub is None:
            return

        # Slightly offset the goal (fixed 0.2 m in Y of map frame) so the target cell is not exactly the same
        # as robot1's footprint center. This reduces immediate collision cost that can make planning fail.
        offset = self.opponent_goal_offset
        msg = PoseStamped()
        msg.header.frame_id = self.opponent_frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = goal_x - offset
        msg.pose.position.y = goal_y - offset
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.opponent_goal_pub.publish(msg)

    def close(self):
        """Placeholder for API compatibility with rsl_rl runner eval scripts."""
        pass

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
        self.goal_marker_pub = rospy.Publisher(f"/{self.ns}/rl_goal_marker", Marker, queue_size=1, latch=True)
        
        self.scan_dim = 360
        # Initialize scan with safe values (e.g. 10.0) to avoid immediate false collision detection (0.0 < threshold)
        self.scan = np.full(self.scan_dim, 10.0)
        self.odom = None
        
        rospy.Subscriber(f"/{self.ns}/scan", LaserScan, self.scan_cb)
        rospy.Subscriber(f"/{self.ns}/odom", Odometry, self.odom_cb)
        
        self.lock = threading.Lock()
        
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.prev_dist_to_goal = 0.0
        self.prev_pose = np.zeros(2)
        
        self.current_action = np.zeros(2)
        self.prev_action = np.zeros(2)

    def scan_cb(self, msg):
        with self.lock:
            scan = np.array(msg.ranges)
            if len(scan) > self.scan_dim:
                scan = scan[:self.scan_dim]
            elif len(scan) < self.scan_dim:
                scan = np.pad(scan, (0, self.scan_dim-len(scan)), 'constant')
            self.scan = np.nan_to_num(scan, posinf=10.0, neginf=0.0)
            # print("scan:@@@@@@@@", self.scan)

    def odom_cb(self, msg):
        with self.lock:
            self.odom = msg
            
    def wait_for_odom(self, timeout=10.0):
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while self.odom is None and not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn(f"[{self.ns}] No odom received within {timeout}s. Verify Gazebo simulation is running.")
                break
            rate.sleep()

    def get_observation(self):
        with self.lock:
            if self.odom is None:
                # Return zero observation if odom is not yet available to prevent crash
                return np.zeros(self.env_cfg.get('num_observations'), dtype=np.float32)

            px = 0.0
            py = 0.0
            yaw = 0.0
            lin_vel = 0.0
            ang_vel = 0.0
            target_dist = 0.0
            target_angle = 0.0
            yaw_err_to_target = 0.0
            
            if self.odom is not None:
                px = self.odom.pose.pose.position.x
                py = self.odom.pose.pose.position.y
                yaw = self.get_yaw(self.odom.pose.pose.orientation)
                # print("yaw:@@@@@@@@", yaw)
                
                lin_vel = self.odom.twist.twist.linear.x
                ang_vel = self.odom.twist.twist.angular.z
                
                dx = self.goal_x - px
                dy = self.goal_y - py
                target_dist = math.sqrt(dx**2 + dy**2)
                
                target_angle = math.atan2(dy, dx) - yaw
                target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))

                yaw_err_to_target = self.goal_yaw - yaw
                yaw_err_to_target = math.atan2(math.sin(yaw_err_to_target), math.cos(yaw_err_to_target))

                # Pose: [x, y, yaw, vx, wz]
                pose_vec = np.array([px, py, yaw, lin_vel, ang_vel], dtype=np.float32)
                
                # Extra: [dist, target_angle, yaw_err_to_target]
                extra = np.array([target_dist, target_angle], dtype=np.float32)

                obs = np.concatenate([
                    pose_vec,
                    self.scan / 10.0,  # normalize lidar ranges（max range 100.0）
                    extra
                ])
                
                # Sanity check: handle NaNs/Infs to prevent PPO explosion
                obs = np.nan_to_num(obs, posinf=10.0, neginf=-10.0)
                obs = np.clip(obs, -20.0, 20.0) # Conservative clip

                if self.env_cfg.get('reward_debug', False):
                    print("--- Observation Debug ---")
                    print('pose_vec:', pose_vec)
                    print('extra:', extra)
                return obs

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def set_action(self, action):
        self.prev_action = self.current_action
        self.current_action = action

        # Apply scaling (PPO output is typically [-1, 1])
        scale = self.env_cfg.get('action_scale')
        scaled_action = action * np.array(scale)

        clip_low = self.env_cfg.get('action_clip_low')
        clip_high = self.env_cfg.get('action_clip_high')
        
        msg = Twist()
        msg.linear.x = np.clip(scaled_action[0], clip_low[0], clip_high[0])
        msg.angular.z = np.clip(scaled_action[1], clip_low[1], clip_high[1])
        
        if self.env_cfg.get('reward_debug', False):
             print("--- Action Debug ---")
             print(f"[set_action] Raw: [{action[0]:.2f}, {action[1]:.2f}] -> Scaled: [{scaled_action[0]:.2f}, {scaled_action[1]:.2f}] -> Clipped: [{msg.linear.x:.2f}, {msg.angular.z:.2f}]")

        self.cmd_vel_pub.publish(msg)



    def compute_reward_and_done(self):
        reward = 0.0
        done = False

        # Configs
        r_cfg = self.reward_cfg
        t_cfg = self.term_cfg

        # State
        # If odom not received yet, skip reward/termination to avoid None access
        if self.odom is None:
            return reward, done

        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        pz = self.odom.pose.pose.position.z
        yaw = self.get_yaw(self.odom.pose.pose.orientation)
        lin_vel = self.odom.twist.twist.linear.x
        ang_vel = self.odom.twist.twist.angular.z
        
        dist_to_goal = math.sqrt((self.goal_x - px)**2 + (self.goal_y - py)**2)
        
        # Target Angle (Relative Bearing)
        dx = self.goal_x - px
        dy = self.goal_y - py
        target_angle = math.atan2(dy, dx) - yaw
        target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle)) # Normalize

        # Yaw Error to Target Goal Orientation
        yaw_err_to_target = self.goal_yaw - yaw
        yaw_err_to_target = math.atan2(math.sin(yaw_err_to_target), math.cos(yaw_err_to_target))

        # --- Reward Calculation ---
        
        # 1. Progress Reward (Dense reward for moving towards goal)
        progress_reward_scale = r_cfg.get('progress_reward_scale')
        # progress = self.prev_dist_to_goal - dist_to_goal
        # progress_reward = progress * progress_reward_scale
        # reward += progress_reward
        progress = lin_vel * math.cos(target_angle)
        dt = self.env_cfg.get('control_dt', 0.1)
        progress_reward = progress * dt * progress_reward_scale
        reward += progress_reward

        # 1.1 Distance Penalty (Potential field to guide globally)
        dist_penalty_scale = r_cfg.get('dist_penalty_scale')
        dist_penalty = dist_to_goal * dist_penalty_scale
        reward += dist_penalty
        
        # 2. Step Cost
        step_cost = r_cfg.get('step_cost')
        reward += step_cost
        
        # 3. Smoothness
        # smoothness_scale is negative, penalizing difference
        diff = self.current_action - self.prev_action
        smoothness_reward = np.dot(diff, diff) * r_cfg.get('smoothness_scale')
        reward += smoothness_reward

        # 3.1 Action Magnitude Penalty (Prevent saturation)
        # Penalize large raw actions to keep them within reasonable range [-1, 1]
        action_penalty = np.sum(np.square(self.current_action)) * r_cfg.get('action_penalty_scale')
        reward += action_penalty
        
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
             diff = min_range_threshold - min_scan
             min_range_reward = r_cfg.get('min_range_penalty') * (math.exp(diff/min_range_threshold) - 1.0)
             reward += min_range_reward

        # 7. Goal & Success
        # Simplified: Reach goal position -> Done + Reward
        at_goal_pos = dist_to_goal < t_cfg.get('success_pos')
        
        goal_reward = r_cfg.get('goal_reward')
        current_goal_reward = 0.0
        
        if at_goal_pos:
            done = True
            current_goal_reward = goal_reward
            # print(f"Robot {self.id} reached goal!")
        
        reward += current_goal_reward
            
        # Termination: Min Height
        # if pz < t_cfg.get('min_height'):
        #     done = True
            
        # Update prev
        self.prev_dist_to_goal = dist_to_goal
        self.prev_pose = np.array([px, py])
        
        if self.env_cfg.get('reward_debug', False):
            print("--- Reward Debug ---")
            print(
                f"Robot {self.id} Total Reward: {reward:.4f}\n"
                f"  Progress: {progress_reward:.4f} (diff: {progress:.4f})\n"
                f"  DistPenalty: {dist_penalty:.4f} dist_to_goal: {dist_to_goal:.4f}\n"
                f"  Step: {step_cost:.4f}\n"
                f"  Smooth: {smoothness_reward:.4f}\n"
                f"  Action: {action_penalty:.4f}\n"
                f"  Head: {heading_penalty:.4f} target_angle: {target_angle:.4f}\n"
                f"  Coll: {collision_reward:.4f}\n"
                f"  MinDistReward: {min_range_reward:.4f} min_scan: {min_scan:.4f}\n"
                f"  Goal: {current_goal_reward:.4f}\n"
                f"  Done: {done}\n"
                f"-------------------"
            )
        
        return float(reward), done

    def reset(self, goal_x, goal_y, goal_yaw, current_x, current_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_yaw = goal_yaw
        
        # Reset state variables
        self.prev_pose = np.array([current_x, current_y])
        self.prev_dist_to_goal = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        
        self.current_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        
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
