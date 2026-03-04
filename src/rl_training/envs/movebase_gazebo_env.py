import math
import threading
import numpy as np
import rospy
import tf.transformations
import torch
from geometry_msgs.msg import PoseStamped, Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry, OccupancyGrid
import sys
import os.path as osp
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tensordict import TensorDict
from visualization_msgs.msg import Marker

sys.path.insert(0, osp.normpath(osp.join(osp.dirname(__file__), '..', 'third_party', 'rsl_rl')))
from rsl_rl.env import VecEnv

class MoveBaseGazeboEnv(VecEnv):
    def __init__(self, env_cfg, device="cpu"):
        self.env_cfg = env_cfg
        self.cfg = env_cfg
        self.device = device

        self.agent_names = env_cfg.get("agent_names", ["robot1"])
        self.num_envs = len(self.agent_names)
        self.num_obs = env_cfg.get("num_observations")
        self.num_actions = env_cfg.get("num_actions")
        self.control_dt = env_cfg.get("control_dt", 0.1)
        self.step_dt = self.control_dt
        self.unwrapped = self

        if not rospy.get_node_uri():
            rospy.init_node("movebase_rl_env", anonymous=True)

        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")

        self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.pause_physics_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_physics_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.reset_world_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.random_spawn_cfg = env_cfg.get("random_spawn", {})
        self.debug = env_cfg.get("debug", False)
        self.ego_spawn_radius = self.random_spawn_cfg.get("ego_spawn_radius", [3.0, 6.0])
        self.goal_reached_dist = env_cfg.get("goal_reached_dist", 0.3)
        self.collision_dist = env_cfg.get("collision_dist", 0.35)

        # Curriculum learning: goal sampling window grows over time (mirrors ros_gazebo_env)
        curriculum_cfg = env_cfg.get("curriculum", {})
        self.goal_span_upper = float(curriculum_cfg.get("initial_span", 4.0))
        self.goal_span_lower = -self.goal_span_upper
        self.goal_span_max = float(curriculum_cfg.get("max_span", 10.0))
        self.goal_span_delta = float(curriculum_cfg.get("delta", 0.004))

        # Map subscription for spawn/goal validity checks (use raw map instead of inflated costmap)
        self.map_topic = env_cfg.get("map_topic", "/map")
        self.map_grid = None
        self.map_lock = threading.Lock()
        self.obstacle_threshold = int(env_cfg.get("map_obstacle_threshold", 50))
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_cb, queue_size=1)

        # Optional opponent (e.g., robot2) goal publishing (only marker/goal, no cmd control)
        opponent_cfg = env_cfg.get("opponent", {})
        self.opponent_enabled = opponent_cfg.get("enabled", False)
        self.opponent_name = opponent_cfg.get("model_name", "robot2")
        self.opponent_goal_topic = opponent_cfg.get("goal_topic", f"/{self.opponent_name}/move_base_simple/goal")
        self.opponent_frame_id = opponent_cfg.get("frame_id", "map")
        self.opponent_goal_offset = opponent_cfg.get("goal_offset", 0.5)
        self.opponent_spawn_radius = opponent_cfg.get("spawn_radius", [1.0, 3.0])
        self.opponent_goal_pub = None
        self.opponent_cmd_pub = None
        if self.opponent_enabled:
            self.opponent_goal_pub = rospy.Publisher(self.opponent_goal_topic, PoseStamped, queue_size=1)
            self.opponent_cmd_pub = rospy.Publisher(f"/{self.opponent_name}/cmd_vel", Twist, queue_size=1)

        self.robots = []
        init_poses = env_cfg.get("init_poses", {})
        for name in self.agent_names:
            self.robots.append(
                MoveBaseRobot(
                    namespace=name,
                    env_cfg=env_cfg,
                    init_pos=init_poses.get(name, [0.0, 0.0, 0.0]),
                )
            )

        rospy.loginfo("Waiting for odom for all robots...")
        for robot in self.robots:
            robot.wait_for_odom()
        rospy.loginfo("MoveBaseGazeboEnv ready.")

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
        self.rew_buf = torch.zeros(self.num_envs, device=device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self.max_episode_length = env_cfg.get("max_episode_length", 500)
        self.episode_length_buf = torch.zeros((self.num_envs, 1), dtype=torch.long, device=device)

    def get_observations(self):
        for i, robot in enumerate(self.robots):
            obs = robot.get_observation()
            self.obs_buf[i] = torch.tensor(obs, device=self.device)
        return TensorDict({"policy": self.obs_buf, "privileged": self.obs_buf.clone()}, batch_size=[self.num_envs])

    def step(self, actions):
        actions_np = actions.detach().cpu().numpy()
        for i, robot in enumerate(self.robots):
            robot.set_action(actions_np[i])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_srv()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")


        rospy.sleep(self.control_dt)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics_srv()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        self.rew_buf[:] = 0.0
        self.reset_buf[:] = False
        self.episode_length_buf += 1

        for i, robot in enumerate(self.robots):
            rew, done = robot.compute_reward_and_done(actions_np[i], self.goal_reached_dist, self.collision_dist)
            self.rew_buf[i] = torch.tensor(rew, dtype=self.rew_buf.dtype, device=self.rew_buf.device)
            self.reset_buf[i] = torch.tensor(done, dtype=self.reset_buf.dtype, device=self.reset_buf.device)

        obs = self.get_observations()
        return obs, self.rew_buf, self.reset_buf, {}

    def reset(self):
        # Full world reset to clear residual dynamics, matching velodyne_env behavior
        if self.debug:
            rospy.loginfo("[env] reset() start")

        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_srv()
            if self.debug:
                rospy.loginfo("[env] reset_world_srv done")

        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        for i in range(self.num_envs):
            self.reset_robot(i)
        if self.debug:
            rospy.loginfo("[env] reset_robot done")

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_srv()
            if self.debug:
                rospy.loginfo("[env] unpause done")
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        rospy.sleep(self.control_dt)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics_srv()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        self.reset_buf[:] = False
        self.episode_length_buf[:] = 0
        return self.get_observations()

    def reset_robot(self, idx):
        robot = self.robots[idx]

        if not robot.spawned_once:
            # First episode: use init_pos directly
            init_x, init_y, init_yaw = robot.init_pos
            gx, gy = init_x + 2.0, init_y
            robot.spawned_once = True
        else:
            ref_x, ref_y, _ = robot.init_pos
            gx, gy = self._curriculum_sample_goal(ref_x, ref_y)
            init_x, init_y = self._sample_around(gx, gy, self.ego_spawn_radius)
            init_yaw = np.random.uniform(-math.pi, math.pi)

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
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        try:
            self.set_model_state_srv(state)
        except rospy.ServiceException as e:
            rospy.logwarn(f"[env] reset_robot set_model_state failed: {e}")

        gyaw = math.atan2(gy - init_y, gx - init_x)
        robot.set_absolute_goal(gx, gy, gyaw)

        # Set opponent goal ONCE per episode, not every step
        if self.opponent_enabled and robot.model_name == self.agent_names[0]:
            self.reset_opponent(gx, gy)
            self.publish_opponent_goal(gx, gy)

    def _curriculum_sample_goal(self, ref_x, ref_y, max_attempts=100):
        """Sample goal within expanding curriculum window relative to ref point."""
        if self.goal_span_upper < self.goal_span_max:
            self.goal_span_upper += self.goal_span_delta
            self.goal_span_lower -= self.goal_span_delta
            if self.debug:
                rospy.loginfo_throttle(10.0, f"[curriculum] goal_span={self.goal_span_upper:.2f}")
        for _ in range(max_attempts):
            gx = ref_x + np.random.uniform(self.goal_span_lower, self.goal_span_upper)
            gy = ref_y + np.random.uniform(self.goal_span_lower, self.goal_span_upper)
            if self._check_pos(gx, gy):
                return gx, gy
        return gx, gy

    def _sample_around(self, cx, cy, radius_range, max_attempts=50):
        """Sample a collision-free position at [r_min, r_max] from (cx, cy)."""
        r_min, r_max = radius_range
        for _ in range(max_attempts):
            r = np.random.uniform(r_min, r_max)
            angle = np.random.uniform(-math.pi, math.pi)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            if self._check_pos(x, y):
                return x, y
        return x, y

    def reset_opponent(self, goal_x, goal_y):
        # Stop robot2 before teleporting to avoid residual velocity
        if self.opponent_cmd_pub is not None:
            self.opponent_cmd_pub.publish(Twist())

        ox, oy = self._sample_around(goal_x, goal_y, self.opponent_spawn_radius)
        oyaw = np.random.uniform(-math.pi, math.pi)

        state = ModelState()
        state.model_name = self.opponent_name
        state.pose.position.x = ox
        state.pose.position.y = oy
        state.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, oyaw)
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        self.set_model_state_srv(state)
        rospy.loginfo(f"[env] Opponent spawned at ({ox:.2f}, {oy:.2f}), goal=({goal_x:.2f}, {goal_y:.2f})")

    def publish_opponent_goal(self, goal_x, goal_y):
        if not self.opponent_enabled or self.opponent_goal_pub is None:
            return

        msg = PoseStamped()
        msg.header.frame_id = self.opponent_frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = goal_x - self.opponent_goal_offset
        msg.pose.position.y = goal_y - self.opponent_goal_offset
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.opponent_goal_pub.publish(msg)

    def close(self):
        pass

    def map_cb(self, msg):
        with self.map_lock:
            self.map_grid = msg

    def _is_free_in_map(self, x, y, threshold=None, check_radius=0.0):
        # Query raw map occupancy grid; treat unknown (-1) and >= threshold as obstacles.
        with self.map_lock:
            mg = self.map_grid
        if mg is None:
            return None
        res = mg.info.resolution
        origin_x = mg.info.origin.position.x
        origin_y = mg.info.origin.position.y
        width = mg.info.width
        height = mg.info.height
        mx = int((x - origin_x) / res)
        my = int((y - origin_y) / res)
        
        thr = self.obstacle_threshold if threshold is None else int(threshold)
        
        # Check a region around the point if check_radius is provided
        range_steps = 0
        if check_radius > 0:
            range_steps = int(math.ceil(check_radius / res))

        for dx in range(-range_steps, range_steps + 1):
            for dy in range(-range_steps, range_steps + 1):
                # Optional: Check circular radius instead of square box
                # if math.sqrt(dx*dx + dy*dy) * res > check_radius:
                #     continue

                nx = mx + dx
                ny = my + dy

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    if self.debug and range_steps == 0:
                        rospy.logwarn_throttle(1.0, f"[env:map] ({x:.2f}, {y:.2f}) out of bounds (w={width}, h={height})")
                    return False
                
                idx = ny * width + nx
                val = mg.data[idx]
                
                if val == -1:
                    if self.debug and range_steps == 0:
                        rospy.logwarn_throttle(1.0, f"[env:map] ({x:.2f}, {y:.2f}) unknown (-1) treated as obstacle")
                    return False
                if val >= thr:
                    if self.debug and range_steps == 0:
                        rospy.logwarn_throttle(1.0, f"[env:map] ({x:.2f}, {y:.2f}) occupied val={val}")
                    return False
        return True

    def _check_pos(self, x, y):
        # Use collision_dist to ensure the spawn position has enough clearance
        free = self._is_free_in_map(x, y, check_radius=self.collision_dist)
        if free is None:
            if self.debug:
                rospy.logwarn_throttle(1.0, "[env] map not received yet, treating position as free")
            return True
        return free


class MoveBaseRobot:
    def __init__(self, namespace, env_cfg, init_pos):
        self.ns = namespace
        self.model_name = namespace
        self.init_pos = init_pos
        self.env_cfg = env_cfg
        self.debug = env_cfg.get("debug", False)
        self.reward_cfg = env_cfg.get("reward", {})
        self.term_cfg = env_cfg.get("termination", {})
        self.lidar_beams = int(env_cfg.get("lidar_beams", 20))
        self.scan_dim = env_cfg.get("scan_dim", 180)
        self.scan = np.full(self.scan_dim, 30.0)
        self.odom = None
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.past_distance = None
        self.spawned_once = False

        self.scan_sub = rospy.Subscriber(f"/{self.ns}/scan", LaserScan, self.scan_cb)
        self.odom_sub = rospy.Subscriber(f"/{self.ns}/odom", Odometry, self.odom_cb)
        self.cmd_pub = rospy.Publisher(f"/{self.ns}/cmd_vel", Twist, queue_size=1)
        self.goal_pub = rospy.Publisher(f"/{self.ns}/move_base_simple/goal", PoseStamped, queue_size=1)
        self.goal_marker_pub = rospy.Publisher(f"/{self.ns}/rl_goal_marker", Marker, queue_size=3)
        self.rl_goal_pub = rospy.Publisher(f"/{self.ns}/rl_goal", PoseStamped, queue_size=1)
        self.lock = threading.Lock()

    def scan_cb(self, msg):
        with self.lock:
            scan = np.array(msg.ranges)
            if len(scan) > self.scan_dim:
                scan = scan[: self.scan_dim]
            elif len(scan) < self.scan_dim:
                scan = np.pad(scan, (0, self.scan_dim - len(scan)), "constant")
            self.scan = np.nan_to_num(scan, posinf=30.0, neginf=0.0)


    def odom_cb(self, msg):
        with self.lock:
            self.odom = msg

    def wait_for_odom(self, timeout=10.0):
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while self.odom is None and not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn(f"[{self.ns}] No odom within {timeout}s")
                break
            rate.sleep()

    def set_action(self, action):
        with self.lock:
            if self.odom is None:
                return

            # Direct cmd_vel control to match TD3 env behavior
            vel = Twist()
            vel.linear.x = float(action[0])
            vel.angular.z = float(action[1])
            self.cmd_pub.publish(vel)
            self.last_cmd = np.array([vel.linear.x, vel.angular.z], dtype=np.float32)
            self.publish_markers(self.last_cmd)

    def get_observation(self):
        with self.lock:
            if self.odom is None:
                return np.zeros(self.env_cfg.get("num_observations"), dtype=np.float32)

            px = self.odom.pose.pose.position.x
            py = self.odom.pose.pose.position.y
            yaw = self._get_yaw(self.odom.pose.pose.orientation)
            dx = self.goal_x - px
            dy = self.goal_y - py
            dist_to_goal = math.sqrt(dx ** 2 + dy ** 2)

            # Simplified theta computation using atan2
            goal_yaw = math.atan2(dy, dx)
            theta = goal_yaw - yaw
            
            # Normalize angle to [-pi, pi]
            theta = math.atan2(math.sin(theta), math.cos(theta))

            # Sector-based min-pooling to match original TD3 velodyne_env.py
            # Divide scan into sectors and take the minimum distance in each sector
            chunk_size = max(1, int(self.scan_dim / self.lidar_beams))
            lidar_norm = np.zeros(self.lidar_beams, dtype=np.float32)
            
            for i in range(self.lidar_beams):
                start = i * chunk_size
                end = min(start + chunk_size, self.scan_dim)
                if start < self.scan_dim:
                    segment = self.scan[start:end]
                    min_val = np.min(segment) if len(segment) > 0 else 30.0
                    lidar_norm[i] = np.clip(min_val / 30.0, 0.0, 1.0)
                else:
                    lidar_norm[i] = 1.0

            obs = np.concatenate(
                [
                    lidar_norm.astype(np.float32),
                    np.array([dist_to_goal, theta, self.last_cmd[0], self.last_cmd[1]], dtype=np.float32),
                ]
            ).astype(np.float32)
            if self.debug:
                print("Observation:",  np.array([dist_to_goal, theta, self.last_cmd[0], self.last_cmd[1]], dtype=np.float32))
            return obs

    def compute_reward_and_done(self, action, goal_reached_dist, collision_dist):
        reward = 0.0
        done = False
        if self.odom is None:
            return reward, done

        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        dx = self.goal_x - px
        dy = self.goal_y - py
        dist_to_goal = math.sqrt(dx ** 2 + dy ** 2)
        min_scan = float(np.nan_to_num(np.min(self.scan), posinf=30.0, neginf=0.0))

        goal_reached = dist_to_goal < goal_reached_dist
        collision = min_scan < collision_dist

        if goal_reached:
            reward = 100.0
            done = True
        elif collision:
            reward = -100.0
            done = True
        else:
            def r3(x):
                return 1 - x if x < 1 else 0.0
            reward = action[0]/ 2 - abs(action[1]) / 2 - r3(min_scan) / 2 - 0.1

        if self.debug:
            rospy.loginfo(
                f"dist={dist_to_goal:.2f} min_scan={min_scan:.2f}\n "
                f"cmd=({action[0]:.2f},{action[1]:.2f})\n "
                f"reward={reward:.2f}\n"
                f"R_v={action[0] / 2} R_w={- abs(action[1]) / 2}"
            )

        
        return float(reward), done

    def set_absolute_goal(self, goal_x, goal_y, goal_yaw):
        """Set goal state and publish for visualization (no move_base command)."""
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_yaw = goal_yaw
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.past_distance = None

        # Publish a marker goal for visualization compatibility
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]
        # self.goal_pub.publish(goal)
        self.rl_goal_pub.publish(goal)
        self.publish_markers(np.zeros(2, dtype=np.float32))

    def _get_yaw(self, q):
        orientation_list = [q.x, q.y, q.z, q.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation_list)
        return yaw

    def publish_markers(self, action):
        # Publish goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.type = goal_marker.CYLINDER
        goal_marker.action = goal_marker.ADD
        goal_marker.scale.x = 0.2
        goal_marker.scale.y = 0.2
        goal_marker.scale.z = 0.2
        goal_marker.color.a = 1.0
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.pose.position.x = self.goal_x
        goal_marker.pose.position.y = self.goal_y
        goal_marker.pose.position.z = 0.0
        
        self.goal_marker_pub.publish(goal_marker)
