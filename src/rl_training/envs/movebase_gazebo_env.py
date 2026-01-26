import math
import threading
from warnings import warn
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
from visualization_msgs.msg import Marker, MarkerArray

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
        # TD3-style expanding goal window used for initial goal sampling
        self.goal_span_upper = 4.0
        self.goal_span_lower = -4.0
        self.goal_reached_dist = env_cfg.get("goal_reached_dist", 0.3)
        self.collision_dist = env_cfg.get("collision_dist", 0.35)

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
        self.opponent_init_pose = env_cfg.get("init_poses", {}).get(
            self.opponent_name, opponent_cfg.get("init_pose")
        )
        self.opponent_goal_pub = None
        if self.opponent_enabled:
            self.opponent_goal_pub = rospy.Publisher(self.opponent_goal_topic, PoseStamped, queue_size=1)

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
            if self.opponent_enabled and robot.model_name == self.agent_names[0]:
                self.publish_opponent_goal(robot.goal_x, robot.goal_y)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_srv()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")


        rospy.sleep(self.control_dt)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics_srv()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.rew_buf[:] = 0.0
        self.reset_buf[:] = False
        self.episode_length_buf += 1

        for i, robot in enumerate(self.robots):
            rew, done = robot.compute_reward_and_done(actions_np[i], self.goal_reached_dist, self.collision_dist)
            self.rew_buf[i] = rew
            self.reset_buf[i] = done

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

        except rospy.ServiceException as e:
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
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        rospy.sleep(self.control_dt)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics_srv()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.reset_buf[:] = False
        self.episode_length_buf[:] = 0
        return self.get_observations()

    def reset_robot(self, idx):
        robot = self.robots[idx]
        x_range = self.random_spawn_cfg.get("x_range")
        y_range = self.random_spawn_cfg.get("y_range")

        if robot.spawned_once:
            init_x = np.random.uniform(x_range[0], x_range[1])
            init_y = np.random.uniform(y_range[0], y_range[1])
            init_yaw = np.random.uniform(-math.pi, math.pi)
        else:
            init_x, init_y, init_yaw = robot.init_pos
            robot.spawned_once = True

        # Resample spawn if map grid reports collision
        attempt = 0
        while not self._check_pos(init_x, init_y) and attempt < 30:
            init_x = np.random.uniform(x_range[0], x_range[1])
            init_y = np.random.uniform(y_range[0], y_range[1])
            init_yaw = np.random.uniform(-math.pi, math.pi)
            attempt += 1

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
        gx, gy, gyaw = self.change_goal(init_x, init_y, init_yaw)
        robot.set_absolute_goal(gx, gy, gyaw)


        if self.opponent_enabled and robot.model_name == self.agent_names[0]:
            self.reset_opponent()

    def change_goal(self, current_x, current_y, current_yaw):
        """TD3-style expanding goal window anchored at current pose."""
        if self.goal_span_upper < 9:
            self.goal_span_upper += 0.004
        if self.goal_span_lower > -9:
            self.goal_span_lower -= 0.004
        goal_ok = False
        gx = current_x
        gy = current_y
        iter_cnt = 0
        
        # Constrain goal within workspace bounds if provided
        x_range = self.random_spawn_cfg.get("x_range")
        y_range = self.random_spawn_cfg.get("y_range")

        while not goal_ok:
            gx = current_x + np.random.uniform(self.goal_span_lower, self.goal_span_upper)
            gy = current_y + np.random.uniform(self.goal_span_lower, self.goal_span_upper)
            
            # Hard limit check
            if not (x_range[0] <= gx <= x_range[1] and y_range[0] <= gy <= y_range[1]):
                goal_ok = False
                continue

            goal_ok = self._check_pos(gx, gy)
            print("goal_ok:", goal_ok)
            iter_cnt += 1
            if iter_cnt > 100:
                warn("Failed to sample valid goal after 100 attempts, using last sampled goal.")
                break

        return gx, gy, current_yaw

    def reset_opponent(self):
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
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        self.set_model_state_srv(state)

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

    def _is_free_in_map(self, x, y, threshold=None):
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
        if mx < 0 or my < 0 or mx >= width or my >= height:
            if self.debug:
                rospy.logwarn_throttle(1.0, f"[env:map] ({x:.2f}, {y:.2f}) out of bounds (w={width}, h={height})")
            return False
        idx = my * width + mx
        val = mg.data[idx]
        thr = self.obstacle_threshold if threshold is None else int(threshold)
        if val == -1:
            if self.debug:
                rospy.logwarn_throttle(1.0, f"[env:map] ({x:.2f}, {y:.2f}) unknown (-1) treated as obstacle")
            return False
        if val >= thr:
            if self.debug:
                rospy.logwarn_throttle(1.0, f"[env:map] ({x:.2f}, {y:.2f}) occupied val={val}")
            return False
        return True

    def _check_pos(self, x, y):
        free = self._is_free_in_map(x, y)
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

            # TD3/velodyne_env-style theta computation (no distance guard)
            dot = dx * 1 + dy * 0
            mag1 = math.sqrt(dx * dx + dy * dy)
            mag2 = 1.0
            beta = math.acos(dot / (mag1 * mag2))
            if dy < 0:
                if dx < 0:
                    beta = -beta
                else:
                    beta = 0 - beta
            theta = beta - yaw
            if theta > math.pi:
                theta = math.pi - theta
                theta = -math.pi - theta
            if theta < -math.pi:
                theta = -math.pi - theta
                theta = math.pi - theta

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
            r3 = lambda x: 1 - x if x < 1 else 0.0
            reward = action[0] / 2 - abs(action[1]) / 2 - r3(min_scan) / 2

        if self.debug:
            rospy.loginfo(
                f"dist={dist_to_goal:.2f} min_scan={min_scan:.2f}\n "
                f"cmd=({action[0]:.2f},{action[1]:.2f})\n "
                f"reward={reward:.2f}\n"
                f"v={action[0] / 2} w={- abs(action[1]) / 2}"
            )

        
        return float(reward), done

    def reset_goal(self, init_x, init_y, init_yaw):
        self.goal_x = init_x
        self.goal_y = init_y
        self.goal_yaw = init_yaw
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.past_distance = None

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
        self.publish_markers(np.zeros(2, dtype=np.float32))

    def _get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

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
