import torch
import numpy as np
import rospy
import tf.transformations
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetPlan, GetPlanRequest
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
        self.init_poses = env_cfg.get('init_poses')

        # Pre-plan all global paths
        self.global_plans = {}
        if env_cfg.get('path', {}).get('enabled', False):
            rospy.loginfo("Pre-planning global paths for all goals...")
            # Assuming robot1 is the agent we are planning for
            plan_service_name = f"/{self.agent_names[0]}/move_base/make_plan"
            rospy.wait_for_service(plan_service_name, timeout=5.0)
            plan_service = rospy.ServiceProxy(plan_service_name, GetPlan)
            
            start_pos = self.init_poses.get(self.agent_names[0])
            start_x, start_y, start_yaw = start_pos[0], start_pos[1], start_pos[2]

            for goal in env_cfg.get('goals'):
                goal_x, goal_y, goal_yaw = goal[0], goal[1], goal[2]
                
                req = GetPlanRequest()
                req.start.header.frame_id = "map"
                req.start.pose.position.x = start_x
                req.start.pose.position.y = start_y
                q_start = tf.transformations.quaternion_from_euler(0, 0, start_yaw)
                req.start.pose.orientation.x = q_start[0]
                req.start.pose.orientation.y = q_start[1]
                req.start.pose.orientation.z = q_start[2]
                req.start.pose.orientation.w = q_start[3]

                req.goal.header.frame_id = "map"
                req.goal.pose.position.x = goal_x
                req.goal.pose.position.y = goal_y
                q_goal = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
                req.goal.pose.orientation.x = q_goal[0]
                req.goal.pose.orientation.y = q_goal[1]
                req.goal.pose.orientation.z = q_goal[2]
                req.goal.pose.orientation.w = q_goal[3]

                res = plan_service(req)
                if res.plan.poses:
                    self.global_plans[tuple(goal)] = res.plan
                else:
                    self.global_plans[tuple(goal)] = None
                    assert False, f"Failed to plan path for goal: {goal}"
                    
        for i, name in enumerate(self.agent_names):
            init_pos = self.init_poses.get(name)
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
        self._call_service_safe(self.unpause_physics_srv, "/gazebo/unpause_physics")

        # 3. Wait for control_dt
        rospy.sleep(self.control_dt)
        
        # 4. Pause physics to freeze state
        self._call_service_safe(self.pause_physics_srv, "/gazebo/pause_physics")

        # 5. Compute rewards/dones and handle resets before returning observations
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

        # 6. After any resets, grab fresh observations for the new state
        obs = self.get_observations()

        # 7. Populate extras for PPO (notably time_outs for bootstrapping)
        self.extras = {"time_outs": time_outs.to(self.device)}

        return obs, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        # Unpause to allow state updates during reset
        self._call_service_safe(self.unpause_physics_srv, "/gazebo/unpause_physics")

        for i in range(self.num_envs):
            self.reset_robot(i)
        
        # Wait for sensors to update after reset
        rospy.sleep(self.control_dt)

        # Pause again
        self._call_service_safe(self.pause_physics_srv, "/gazebo/pause_physics")

        self.episode_length_buf[:] = 0
        self.reset_buf[:] = False
        return self.get_observations()

    def _call_service_safe(self, service_proxy, service_name):
        """Call ROS service with error logging."""
        try:
            service_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"{service_name} service call failed: {e}")

    def reset_robot(self, idx):
        robot = self.robots[idx]
        init_x, init_y, init_yaw = robot.init_pos
        
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
        goals = self.env_cfg.get('goals')
        goal_idx = np.random.randint(0, len(goals))
        goal = goals[goal_idx]
        goal_x, goal_y, goal_yaw = goal[0], goal[1], goal[2]
        path_for_goal = self.global_plans.get(tuple(goal))
        robot.reset(goal_x, goal_y, goal_yaw, init_x, init_y, init_yaw, path=path_for_goal)

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

        self._call_service_safe(lambda: self.set_model_state_srv(state), "/gazebo/set_model_state")

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
        self.path_cfg = env_cfg.get('path', {})

        self.plan_publisher = None
        if self.path_cfg.get('enabled') and self.path_cfg.get('visualize'):
            self.plan_publisher = rospy.Publisher(f"/{self.ns}/global_plan", Path, queue_size=1, latch=True)
        
        self.cmd_vel_pub = rospy.Publisher(f"/{self.ns}/cmd_vel", Twist, queue_size=1)
        self.goal_marker_pub = rospy.Publisher(f"/{self.ns}/rl_goal_marker", Marker, queue_size=1, latch=True)
        
        self.global_plan = None
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

        # Path-following related
        self.cumulative_path_lengths = None
        self.prev_path_progress = 0.0


    def scan_cb(self, msg):
        with self.lock:
            scan = np.array(msg.ranges)
            if len(scan) > self.scan_dim:
                scan = scan[:self.scan_dim]
            elif len(scan) < self.scan_dim:
                scan = np.pad(scan, (0, self.scan_dim-len(scan)), 'constant')
            self.scan = np.nan_to_num(scan, posinf=10.0, neginf=0.0)

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
            
            # Use previous action instead of odometry velocity as per user request
            # lin_vel = self.odom.twist.twist.linear.x
            # ang_vel = self.odom.twist.twist.angular.z
            
            dx = self.goal_x - px
            dy = self.goal_y - py
            target_dist = math.sqrt(dx**2 + dy**2)
            
            target_angle = math.atan2(dy, dx) - yaw
            target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))

            # Pose: [x, y, yaw, prev_v_cmd, prev_w_cmd]
            pose_vec = np.array([px, py, yaw, self.prev_action[0], self.prev_action[1]], dtype=np.float32)
            
            # Extra: [dist, target_angle]
            extra = np.array([dx,dy, target_angle], dtype=np.float32)

            obs = np.concatenate([
                pose_vec,
                self.scan / 10.0,  # normalize lidar ranges
                extra
            ])
            
            # Sanity check: handle NaNs/Infs to prevent PPO explosion
            obs = np.nan_to_num(obs, posinf=10.0, neginf=-10.0)
            obs = np.clip(obs, -20.0, 20.0)

            if self.env_cfg.get('reward_debug', False):
                print("--- Observation Debug ---")
                print('pose_vec:', pose_vec)
                print('extra:', extra)
            return obs

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

            
    def calculate_path_errors(self, robot_x, robot_y):
        """
        Calculates both cross-track error and longitudinal progress along the path.
        - Cross-track error (CTE): The minimum perpendicular distance to the path.
        - Progress: The distance from the start of the path to the closest point on the path.
        """
        if self.global_plan is None or not self.global_plan.poses or self.cumulative_path_lengths is None:
            return 0.0, self.prev_path_progress # Return 0 CTE and non-advancing progress

        path_points = np.array([[p.pose.position.x, p.pose.position.y] for p in self.global_plan.poses])
        robot_pos = np.array([robot_x, robot_y])

        min_dist_sq = float('inf')
        best_progress = self.prev_path_progress # Default to previous progress
        
        # Iterate over all path segments
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]

            # Vector of the segment
            line_vec = p2 - p1
            segment_len_sq = np.dot(line_vec, line_vec)

            # If segment has zero length, skip
            if segment_len_sq < 1e-6:
                continue

            # Vector from segment start to robot
            robot_vec = robot_pos - p1

            # Projection of robot_vec onto line_vec, scaled by segment length
            t = np.dot(robot_vec, line_vec) / segment_len_sq
            
            # Clamp t to be within the segment [0, 1]
            t_clamped = np.clip(t, 0.0, 1.0)

            # Find the closest point on the (infinite) line or the segment endpoints
            closest_point_on_segment = p1 + t_clamped * line_vec
            
            # Calculate distance from robot to this closest point
            dist_sq = np.sum((robot_pos - closest_point_on_segment)**2)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                # Progress is the cumulative length to the start of the segment plus the projected length
                segment_progress = t_clamped * np.sqrt(segment_len_sq)
                best_progress = self.cumulative_path_lengths[i] + segment_progress

        cross_track_error = np.sqrt(min_dist_sq)
        
        # Ensure progress is monotonically increasing to avoid rewards for moving backwards
        # This can happen if the robot cuts a corner.
        if best_progress < self.prev_path_progress - 0.1: # Allow for small numerical errors
             return cross_track_error, self.prev_path_progress

        return cross_track_error, best_progress

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

        # If odom not received yet, skip reward/termination to avoid None access
        if self.odom is None:
            return reward, done

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
        
        # Target Angle (Relative Bearing)
        dx = self.goal_x - px
        dy = self.goal_y - py
        target_angle = math.atan2(dy, dx) - yaw
        target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))

        # Yaw Error to Target Goal Orientation
        yaw_err_to_target = self.goal_yaw - yaw
        yaw_err_to_target = math.atan2(math.sin(yaw_err_to_target), math.cos(yaw_err_to_target))

        # --- Reward Calculation ---
        
        # 1. Progress Reward (Dense reward for moving towards goal)
        progress_reward_scale = r_cfg.get('progress_reward_scale')
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
        diff = self.current_action - self.prev_action
        smoothness_reward = np.dot(diff, diff) * r_cfg.get('smoothness_scale')
        reward += smoothness_reward

        # 3.1 Action Magnitude Penalty (Prevent saturation)
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
        is_final_goal_reached = dist_to_goal < t_cfg.get('success_pos')
        
        current_goal_reward = 0.0
        if is_final_goal_reached:
            done = True
            current_goal_reward = r_cfg.get('goal_reward')
            rospy.loginfo(f"Robot {self.id} reached final goal!")
        
        reward += current_goal_reward
            
        # 8. Path Following Rewards
        path_progress_reward = 0.0
        cross_track_penalty = 0.0
        cross_track_error = 0.0
        progress_delta = 0.0  # Initialize here to prevent UnboundLocalError
        if self.path_cfg.get('enabled', False) and self.global_plan is not None:
            # Calculate both CTE and progress along path
            cross_track_error, current_path_progress = self.calculate_path_errors(px, py)

            # A. Cross-Track Error Penalty (Lateral Control)
            # Penalize the robot for being far from the path.
            cte_penalty_scale = self.path_cfg.get('cross_track_penalty_scale')
            cross_track_penalty = cte_penalty_scale * cross_track_error
            reward += cross_track_penalty

            # B. Path Progress Reward (Longitudinal Control), like MetaDrive's driving_reward
            progress_reward_scale = self.path_cfg.get('path_progress_reward_scale')
            progress_delta = current_path_progress - self.prev_path_progress
            path_progress_reward = progress_reward_scale * progress_delta
            reward += path_progress_reward

            # Update previous progress for next step
            self.prev_path_progress = current_path_progress

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
                f"  CrossTrackPenalty: {cross_track_penalty:.4f} (cte: {cross_track_error:.4f})\n"
                f"  PathProgressReward: {path_progress_reward:.4f} (progress_delta: {progress_delta:.4f})\n"
                f"  Goal: {current_goal_reward:.4f}\n"
                f"  Done: {done}\n"
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
        marker.pose.position.z = 0.5
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
        
        # Set the pre-planned global path
        if path is None:
            assert False, f"[{self.ns}] No pre-planned path provided for goal ({goal_x}, {goal_y})"
        self.global_plan = path
        self.cumulative_path_lengths = None # Reset and recalculate

        if self.global_plan and self.global_plan.poses:
            # Manually set the header frame_id for RViz visualization
            self.global_plan.header.frame_id = "map"
            self.global_plan.header.stamp = rospy.Time.now()

            # Pre-calculate cumulative path lengths for efficient progress calculation
            path_points = np.array([[p.pose.position.x, p.pose.position.y] for p in self.global_plan.poses])
            segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
            self.cumulative_path_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
            
            if self.path_cfg.get('visualize', False) and self.plan_publisher is not None:
                self.plan_publisher.publish(self.global_plan)
        
        # Reset state variables
        self.prev_pose = np.array([current_x, current_y])
        self.prev_path_progress = 0.0 # Reset progress at the start of a new episode
        if self.goal_x is not None:
            self.prev_dist_to_goal = math.sqrt((self.goal_x - current_x)**2 + (self.goal_y - current_y)**2)
        else:
            self.prev_dist_to_goal = 0.0
        
        self.current_action = np.zeros(2)
        self.prev_action = np.zeros(2)
