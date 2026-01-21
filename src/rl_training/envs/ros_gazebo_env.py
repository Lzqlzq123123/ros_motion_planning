import torch
import numpy as np
import rospy
import tf.transformations
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
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
        self.cfg = env_cfg  # Alias for runner logging compatibility
        self.device = device
        
        # Extract robot configurations
        self.agent_names = env_cfg.get('agent_names')
        self.num_envs = len(self.agent_names)
        
        # Observation and Action spaces
        self.num_obs = env_cfg.get('num_observations')
        self.num_actions = env_cfg.get('num_actions')
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
        self.init_poses = env_cfg.get('init_poses')

        # Goal sampling
        self.goal_radius_range = env_cfg.get('goal_radius_range')
        self.max_attempts = int(env_cfg.get('max_attempts', 50))
        gv_cfg = env_cfg.get('goal_validation', {})
        self.costmap_topic = gv_cfg.get('costmap_topic', '/move_base/global_costmap/costmap')
        self.costmap_free_thresh = gv_cfg.get('costmap_free_threshold', 50)
        self.costmap_inflation = float(gv_cfg.get('costmap_inflation', 0.2))
        self.costmap_data = None
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self._costmap_cb, queue_size=1)
        
        for i, name in enumerate(self.agent_names):
            init_pos = self.init_poses.get(name)
            self.robots.append(RobotAgent(name, i, name, env_cfg, init_pos))

        # Track whether each robot has completed its first spawn (first uses fixed init pose)
        self.first_reset_done = [False] * self.num_envs

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
        self.opponent_goal_offset = opponent_cfg.get('goal_offset')
        self.opponent_init_pose = self.init_poses.get(self.opponent_name, opponent_cfg.get('init_pose'))
        self.opponent_goal_pub = None
        if self.opponent_enabled:
            self.opponent_goal_pub = rospy.Publisher(self.opponent_goal_topic, PoseStamped, queue_size=1)
            
        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.rew_buf = torch.zeros(self.num_envs, device=device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self.max_episode_length = env_cfg.get('max_episode_length', int(1e9))
        self.episode_length_buf = torch.zeros((self.num_envs, 1), dtype=torch.long, device=device)
        

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
        self._call_service_safe(self.unpause_physics_srv, "/gazebo/unpause_physics")

        # 3. Wait for control_dt
        rospy.sleep(self.control_dt)
        
        # 4. Pause physics to freeze state
        self._call_service_safe(self.pause_physics_srv, "/gazebo/pause_physics")

        # 5. Compute rewards/dones and handle resets before returning observations
        self.rew_buf[:] = 0.0
        self.reset_buf[:] = False
        self.episode_length_buf += 1
        
        for i, robot in enumerate(self.robots):
            rew, done = robot.compute_reward_and_done()
            self.rew_buf[i] = rew
            self.reset_buf[i] = done
            
            if done:
                self.episode_length_buf[i] = 0
                self.reset_robot(i)
                
        # 6. After any resets, grab fresh observations for the new state
        obs = self.get_observations()

        # 7. Return extras (empty)
        return obs, self.rew_buf, self.reset_buf, {}

    def reset(self):
        # Unpause to allow state updates during reset
        self._call_service_safe(self.unpause_physics_srv, "/gazebo/unpause_physics")

        for i in range(self.num_envs):
            self.reset_robot(i)
        
        # Wait for sensors to update after reset
        rospy.sleep(self.control_dt)

        # Pause again
        self._call_service_safe(self.pause_physics_srv, "/gazebo/pause_physics")

        self.reset_buf[:] = False
        self.episode_length_buf[:] = 0
        return self.get_observations()

    def _call_service_safe(self, service_proxy, service_name):
        """Call ROS service with error logging."""
        try:
            service_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"{service_name} service call failed: {e}")

    def sample_goal(self, start_x, start_y, start_yaw):
        """Sample a goal around initial position with radius r; optionally reject if costmap shows collision."""
        attempts = max(1, self.max_attempts)
        fallback_goal = None

        for _ in range(attempts):
            # Sample goal within radius range from initial position
            min_radius, max_radius = self.goal_radius_range
            radius = np.random.uniform(min_radius, max_radius)
            angle = np.random.uniform(0, 2 * math.pi)
            
            goal_x = start_x + radius * math.cos(angle)
            goal_y = start_y + radius * math.sin(angle)
            goal_yaw = np.random.uniform(-math.pi, math.pi)

            fallback_goal = (goal_x, goal_y, goal_yaw)

            if self.costmap_data is None:
                # No costmap yet; accept and move on to avoid blocking resets
                return goal_x, goal_y, goal_yaw

            if self._is_free_in_costmap(goal_x, goal_y):
                return goal_x, goal_y, goal_yaw

        rospy.logwarn("Goal sampling hit max attempts; using last sampled goal (may be occupied).")
        if fallback_goal is not None:
            return fallback_goal
        return start_x, start_y, 0.0  # Fallback to initial position

    def _costmap_cb(self, msg):
        """Cache costmap for goal filtering."""
        self.costmap_data = msg

    def _is_free_in_costmap(self, _x, _y):
        grid = self.costmap_data
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        width = grid.info.width
        height = grid.info.height

        inflate_cells = max(0, int(self.costmap_inflation / res))

        cx = int((_x - ox) / res)
        cy = int((_y - oy) / res)

        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return False

        data = grid.data

        def cell_cost(ix, iy):
            idx = iy * width + ix
            if idx < 0 or idx >= len(data):
                return 100
            return data[idx]

        for ix in range(cx - inflate_cells, cx + inflate_cells + 1):
            for iy in range(cy - inflate_cells, cy + inflate_cells + 1):
                if ix < 0 or iy < 0 or ix >= width or iy >= height:
                    return False
                val = cell_cost(ix, iy)
                if val == -1:  # unknown treated as occupied
                    return False
                if val >= self.costmap_free_thresh:
                    return False
        return True

    def sample_free_position(self, x_range, y_range):
        """Sample a free position using costmap validation (similar to sample_goal)."""
        if self.costmap_data is None:
            # No costmap yet; return random position
            return np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])
        
        for _ in range(self.max_attempts):
            init_x = np.random.uniform(x_range[0], x_range[1])
            init_y = np.random.uniform(y_range[0], y_range[1])
            
            if self._is_free_in_costmap(init_x, init_y):
                return init_x, init_y
        
        rospy.logwarn("Position sampling hit max attempts; using last sampled position (may be occupied).")
        return np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])

    def reset_robot(self, idx):
        robot = self.robots[idx]
        # First reset uses configured init pose; subsequent resets sample random pose/yaw
        if not self.first_reset_done[idx]:
            init_x, init_y, init_yaw = robot.init_pos
            self.first_reset_done[idx] = True
        else:
            rand_cfg = self.env_cfg.get('random_spawn')
            x_min, x_max = rand_cfg.get('x_range')
            y_min, y_max = rand_cfg.get('y_range')
            
            # Use costmap validation for random spawn position
            init_x, init_y = self.sample_free_position([x_min, x_max], [y_min, y_max])
            init_yaw = np.random.uniform(-math.pi, math.pi)
        
        # Reset Pose
        state = ModelState()
        state.model_name = robot.model_name
        state.pose.position.x = init_x
        state.pose.position.y = init_y
        state.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, init_yaw)
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

        self._call_service_safe(lambda: self.set_model_state_srv(state), "/gazebo/set_model_state")
        
        # Generate New Goal
        goal_x, goal_y, goal_yaw = self.sample_goal(init_x, init_y, init_yaw)
        robot.reset(goal_x, goal_y, goal_yaw, init_x, init_y, init_yaw, path=None)

        # Track last main goal for opponent reference
        self.latest_goal = (goal_x, goal_y, goal_yaw)

        # Also reset and command the interference robot (e.g., robot2)
        if self.opponent_enabled and robot.model_name == self.agent_names[0]:
            self.reset_interference_robot(goal_x, goal_y)

    def reset_interference_robot(self, goal_x, goal_y):
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

        self._call_service_safe(lambda: self.set_model_state_srv(state), "/gazebo/set_model_state")

        # Send goal for opponent: mirror primary goal with offset
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
        self.path_cfg = env_cfg.get('path', {})

        self.plan_publisher = None
        if self.path_cfg.get('enabled') and self.path_cfg.get('visualize'):
            self.plan_publisher = rospy.Publisher(f"/{self.ns}/global_plan", Path, queue_size=1, latch=True)
        
        self.cmd_vel_pub = rospy.Publisher(f"/{self.ns}/cmd_vel", Twist, queue_size=1)
        self.goal_marker_pub = rospy.Publisher(f"/{self.ns}/rl_goal_marker", Marker, queue_size=1, latch=True)
        
        self.global_plan = None
        self.scan_dim = 180
        # Initialize scan with safe values (e.g. 10.0) to avoid immediate false collision detection (0.0 < threshold)
        self.scan = np.full(self.scan_dim, 10.0)
        self.odom = None
        
        rospy.Subscriber(f"/{self.ns}/scan", LaserScan, self.scan_cb)
        rospy.Subscriber(f"/{self.ns}/odom", Odometry, self.odom_cb)
        
        self.lock = threading.Lock()
        
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.last_cmd = np.zeros(2, dtype=np.float32)
        # Track previous distance to goal for progress-based reward
        self.past_distance = None


    def scan_cb(self, msg):
        with self.lock:
            scan = np.array(msg.ranges)
            if len(scan) > self.scan_dim:
                scan = scan[:self.scan_dim]
            elif len(scan) < self.scan_dim:
                scan = np.pad(scan, (0, self.scan_dim-len(scan)), 'constant')
            self.scan = np.nan_to_num(scan, posinf=30.0, neginf=0.0)

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

            px = self.odom.pose.pose.position.x
            py = self.odom.pose.pose.position.y
            yaw = self.get_yaw(self.odom.pose.pose.orientation)
            yaw_deg = math.degrees(yaw) % 360.0
            dx = self.goal_x - px
            dy = self.goal_y - py
            dist_to_goal = math.sqrt(dx**2 + dy**2)
            goal_heading = math.atan2(dy, dx)  # [-pi, pi]

            # Downsample LiDAR: take 30 evenly spaced beams, then min-pool each 3-beam group -> 10 dims
            sample_indices = np.linspace(0, self.scan_dim - 1, 30).astype(int)
            scan_sampled = self.scan[sample_indices].reshape(10, 3)
            lidar_pooled = np.min(scan_sampled, axis=1) / 30.0  # normalize by assumed max range 30m

            # Normalize yaw to [0, 1], and heading alignment error to [-1, 1]
            yaw_norm = yaw_deg / 360.0
            diff_angle = (yaw_deg - math.degrees(goal_heading))
            diff_angle = (diff_angle + 180.0) % 360.0 - 180.0  # wrap to [-180, 180]
            diff_angle_norm = diff_angle / 180.0

            obs = np.concatenate([
                lidar_pooled.astype(np.float32),
                np.array([dist_to_goal, goal_heading], dtype=np.float32),
                np.array([
                    self.last_cmd[0],
                    self.last_cmd[1],
                    yaw_norm,
                    diff_angle_norm,
                ], dtype=np.float32),
            ]).astype(np.float32)

            if self.env_cfg.get('reward_debug', False):
                print("--- Observation Debug ---")
                print('lidar_pooled:', lidar_pooled)
                print('dist_to_goal:', dist_to_goal)
                print('goal_heading(rad):', goal_heading)
                print('yaw_norm:', yaw_norm, 'diff_angle_norm:', diff_angle_norm)
            return obs

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

            
    def calculate_path_errors(self, robot_x, robot_y):
        return 0.0, 0.0

    def set_action(self, action):
        # Apply scaling (PPO output is typically [-1, 1])
        scale = self.env_cfg.get('action_scale')
        scaled_action = action * np.array(scale)

        clip_low = self.env_cfg.get('action_clip_low')
        clip_high = self.env_cfg.get('action_clip_high')
        
        msg = Twist()
        msg.linear.x = np.clip(scaled_action[0], clip_low[0], clip_high[0])
        msg.angular.z = np.clip(scaled_action[1], clip_low[1], clip_high[1])

        self.last_cmd = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

        if self.env_cfg.get('reward_debug', False):
             print("--- Action Debug ---")
             print(f"[set_action] Raw: [{action[0]:.2f}, {action[1]:.2f}] -> Scaled: [{scaled_action[0]:.2f}, {scaled_action[1]:.2f}] -> Clipped: [{msg.linear.x:.2f}, {msg.angular.z:.2f}]")

        self.cmd_vel_pub.publish(msg)

    def compute_reward_and_done(self):
        reward = 0.0
        done = False

        if self.odom is None:
            return reward, done

        r_cfg = self.reward_cfg
        t_cfg = self.term_cfg

        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        dx = self.goal_x - px
        dy = self.goal_y - py

        dist_to_goal = math.sqrt(dx**2 + dy**2)
        min_scan = float(np.min(self.scan))
        min_collision_range = t_cfg.get('min_collision_range')

        # Read shaping params early so debug printing can always access them
        rs = self.env_cfg.get('reward_shaping', {})
        distance_scale = float(rs.get('distance_scale'))
        wall_scale = float(rs.get('wall_scale'))
        time_step_pen = float(rs.get('time_penalty'))
        diagonal_base = float(rs.get('diagonal_base'))
        heading_scale = float(rs.get('heading_scale'))
        ang_vel_scale = float(rs.get('ang_vel_scale'))

        # debug helpers
        distance_rate = 0.0
        current_pen_dis = 0.0
        max_state = 0.0
        value_middle = 0.0
        wall_rate_pen = 0.0
        distance_component = 0.0
        wall_component = 0.0
        heading_component = 0.0
        ang_vel_component = 0.0
        event_str = "step"

        goal_reached = dist_to_goal < t_cfg.get('success_pos')
        collision = min_scan < min_collision_range

        if goal_reached:
            goal = r_cfg.get('goal_reward')
            reward = goal
            event_str = "goal"
            done = True
            rospy.loginfo(f"Robot {self.id} reached goal. Reward: {reward}")
        elif collision: 
            collision_reward = r_cfg.get('collision_penalty')
            reward = collision_reward
            event_str = "collision"
            done = True
        else:
            # Progress-based reward (distance reduction), wall proximity penalty, and time penalty
            current_distance = dist_to_goal

            # Initialize past_distance if not set
            if self.past_distance is None:
                self.past_distance = current_distance

            # Distance progress
            distance_rate = (self.past_distance - current_distance)
            diagonal = diagonal_base * math.sqrt(2.0)
            if distance_rate >= 0:
                distance_rate = distance_rate * (1.0 + (diagonal - current_distance) / diagonal)
            else:
                distance_rate = distance_rate * (1.0 + current_distance / diagonal)

            # Wall penalty based on scan distribution (replicates pen_wall logic)
            scan_norm = np.clip(self.scan / 30.0, 0.0, 1.0)
            max_state = float(np.max(scan_norm)) if scan_norm.size > 0 else 0.0
            if scan_norm.size % 2 != 0:
                idx_middle = scan_norm.size // 2
                value_middle = float(scan_norm[idx_middle])
            else:
                idx_g = scan_norm.size // 2
                idx_l = idx_g - 1
                value_middle = max(float(scan_norm[idx_g]), float(scan_norm[idx_l]))
            if value_middle < 0.2 * max_state:
                current_pen_dis = (max_state - value_middle)
            else:
                current_pen_dis = 0.0
            
            distance_component = distance_scale * distance_rate
            wall_rate_pen = -current_pen_dis
            wall_component = wall_scale * wall_rate_pen

            reward = distance_component + wall_component + time_step_pen

            # Heading reward: encourage facing goal
            yaw = self.get_yaw(self.odom.pose.pose.orientation)
            goal_heading = math.atan2(dy, dx)
            heading_error = math.atan2(math.sin(goal_heading - yaw), math.cos(goal_heading - yaw))
            heading_component = heading_scale * math.cos(heading_error)

            # Angular velocity penalty: discourage large spins with quadratic penalty
            # Use normalized angular velocity to make penalty more effective
            normalized_angular_vel = abs(self.last_cmd[1]) / 0.5  # Normalize to [0, 1] range
            ang_vel_component = ang_vel_scale * (normalized_angular_vel ** 2)

            reward += heading_component + ang_vel_component

            # Update past distance for next step
            self.past_distance = current_distance

        if self.env_cfg.get('reward_debug', False):
            print("--- Reward Debug ---")
            print(
                f"Event: {event_str} | Total Reward: {reward:.4f}\n"
                f"  current_distance: {dist_to_goal:.3f}\n"
                f"  distance_rate: {distance_rate:.6f} | distance_component: {distance_component:.4f}\n"
                f"  wall: max_state={max_state:.3f}, middle={value_middle:.3f}, current_pen={current_pen_dis:.4f}| wall_component: {wall_component:.4f}\n"
                f"  heading_component: {heading_component:.4f} | ang_vel_component: {ang_vel_component:.4f}\n"
                f"  time_penalty: {time_step_pen:.3f}\n"
                f"  min_scan: {min_scan:.3f} | cmd: [{self.last_cmd[0]:.3f}, {self.last_cmd[1]:.3f}]\n"
                f"  goal_reached: {goal_reached}, collision: {collision}\n"
                f"-------------------"
            )

        return float(reward), done


    def publish_goal_marker(self, goal_x, goal_y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rl_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal_x
        marker.pose.position.y = goal_y
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.goal_marker_pub.publish(marker)

    def reset(self, goal_x, goal_y, goal_yaw, current_x, current_y, current_yaw, path=None):
        # Set goal
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_yaw = goal_yaw
        if self.goal_x is not None:
            self.publish_goal_marker(self.goal_x, self.goal_y)
        
        # 路径可选：若未提供全局路径则清空并继续（与 velodyne 对齐无需路径）
        self.global_plan = path
        if self.global_plan and self.global_plan.poses:
            self.global_plan.header.frame_id = "map"
            self.global_plan.header.stamp = rospy.Time.now()
            if self.path_cfg.get('visualize', False) and self.plan_publisher is not None:
                self.plan_publisher.publish(self.global_plan)
        
        # Reset state variables
        self.last_cmd = np.zeros(2, dtype=np.float32)
        # Initialize past distance used by progress-based reward
        try:
            self.past_distance = math.hypot(self.goal_x - current_x, self.goal_y - current_y)
        except Exception:
            self.past_distance = None
