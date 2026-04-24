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


class RuleBasedAdversary:
    """Rule-based adversarial controller for opponent vehicle (robot2).

    Three scripted attack behaviors are randomly selected per episode:
      - head_on:   Drive straight toward the ego robot to force a head-on collision.
      - cross:     Move perpendicular to ego's heading to simulate T-intersection crossing.
      - follow_stop: Follow the ego robot closely, then suddenly brake.
    """

    BEHAVIORS = ["head_on", "cross", "follow_stop"]

    def __init__(self, opponent_name, ego_name, cfg):
        self.opponent_name = opponent_name
        self.ego_name = ego_name

        # Tuneable parameters
        self.linear_speed = float(cfg.get("linear_speed", 0.5))
        self.angular_gain = float(cfg.get("angular_gain", 2.0))
        self.follow_dist = float(cfg.get("follow_dist", 1.0))
        self.stop_prob = float(cfg.get("stop_prob", 0.1))

        # ROS publishers / subscribers
        self.cmd_pub = rospy.Publisher(
            f"/{opponent_name}/cmd_vel", Twist, queue_size=1
        )
        self.ego_odom = None
        self.adv_odom = None
        self._ego_lock = threading.Lock()
        self._adv_lock = threading.Lock()

        self.ego_odom_sub = rospy.Subscriber(
            f"/{ego_name}/odom", Odometry, self._ego_odom_cb, queue_size=1
        )
        self.adv_odom_sub = rospy.Subscriber(
            f"/{opponent_name}/odom", Odometry, self._adv_odom_cb, queue_size=1
        )

        # Current episode behavior
        self.behavior = "head_on"
        self._follow_braking = False

    # ---- ROS callbacks ----
    def _ego_odom_cb(self, msg):
        with self._ego_lock:
            self.ego_odom = msg

    def _adv_odom_cb(self, msg):
        with self._adv_lock:
            self.adv_odom = msg

    # ---- Episode lifecycle ----
    def on_reset(self):
        """Called at the beginning of each episode to pick a new behavior."""
        self.behavior = np.random.choice(self.BEHAVIORS)
        self._follow_braking = False
        # Stop opponent immediately
        self.cmd_pub.publish(Twist())

    # ---- Main control tick (called once per env.step) ----
    def step(self):
        """Compute and publish a cmd_vel for the adversary."""
        with self._ego_lock:
            ego = self.ego_odom
        with self._adv_lock:
            adv = self.adv_odom

        if ego is None or adv is None:
            return

        ex = ego.pose.pose.position.x
        ey = ego.pose.pose.position.y
        ax = adv.pose.pose.position.x
        ay = adv.pose.pose.position.y

        dx = ex - ax
        dy = ey - ay
        dist = math.sqrt(dx * dx + dy * dy)
        angle_to_ego = math.atan2(dy, dx)

        adv_yaw = self._yaw(adv.pose.pose.orientation)

        if self.behavior == "head_on":
            cmd = self._head_on(angle_to_ego, adv_yaw)
        elif self.behavior == "cross":
            cmd = self._cross(angle_to_ego, adv_yaw)
        elif self.behavior == "follow_stop":
            cmd = self._follow_stop(angle_to_ego, adv_yaw, dist)
        else:
            cmd = Twist()

        self.cmd_pub.publish(cmd)

    # ---- Behavior implementations ----
    def _head_on(self, angle_to_ego, adv_yaw):
        """Drive directly toward ego at full speed."""
        cmd = Twist()
        err = self._angle_diff(angle_to_ego, adv_yaw)
        cmd.linear.x = self.linear_speed
        cmd.angular.z = np.clip(self.angular_gain * err, -1.0, 1.0)
        return cmd

    def _cross(self, angle_to_ego, adv_yaw):
        """Move perpendicular to the ego-adv line (T-intersection crossing)."""
        cmd = Twist()
        perp_angle = angle_to_ego + math.pi / 2.0
        err = self._angle_diff(perp_angle, adv_yaw)
        cmd.linear.x = self.linear_speed
        cmd.angular.z = np.clip(self.angular_gain * err, -1.0, 1.0)
        return cmd

    def _follow_stop(self, angle_to_ego, adv_yaw, dist):
        """Follow ego; with probability stop_prob per tick, execute sudden brake."""
        cmd = Twist()
        if self._follow_braking:
            # Stay stopped for the rest of this episode
            return cmd
        # Random sudden stop
        if dist < self.follow_dist and np.random.random() < self.stop_prob:
            self._follow_braking = True
            return cmd
        # Chase ego
        err = self._angle_diff(angle_to_ego, adv_yaw)
        cmd.linear.x = self.linear_speed
        cmd.angular.z = np.clip(self.angular_gain * err, -1.0, 1.0)
        return cmd

    # ---- Helpers ----
    @staticmethod
    def _yaw(q):
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    @staticmethod
    def _angle_diff(target, current):
        d = target - current
        return math.atan2(math.sin(d), math.cos(d))

class MoveBaseGazeboEnv(VecEnv):
    def __init__(self, env_cfg, device="cpu"):
        self.env_cfg = env_cfg
        self.cfg = env_cfg
        self.device = device

        # Control mode for robot1: 'td3' (RL policy) or 'movebase' (traditional planner)
        self.robot1_mode = env_cfg.get("robot1_mode", "td3")

        self.agent_names = env_cfg.get("agent_names", ["robot1"])
        self.num_envs = len(self.agent_names)
        self.num_obs = env_cfg.get("num_observations")
        self.num_actions = env_cfg.get("num_actions")
        self.control_dt = env_cfg.get("control_dt", 0.1)
        self.step_dt = self.control_dt
        self.unwrapped = self
        self.fixed_episode_setup = None
        self.last_reset_info = {}

        if not rospy.get_node_uri():
            rospy.init_node("movebase_rl_env", anonymous=True)

        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")

        self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.pause_physics_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_physics_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.reset_world_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.debug = env_cfg.get("debug", False)
        self.goal_reached_dist = env_cfg.get("goal_reached_dist", 0.3)
        self.collision_dist = env_cfg.get("collision_dist", 0.35)
        self.ego_spawn_radius = env_cfg.get("ego_spawn_radius", [3.0, 6.0])
        self.adv_spawn_radius = env_cfg.get("adv_spawn_radius", [1.0, 3.0])
        self.goal_mode = env_cfg.get("goal_mode", "random")
        goal_range = env_cfg.get("goal_range", {})
        self.goal_x_min = float(goal_range.get("x_min", -8.0))
        self.goal_x_max = float(goal_range.get("x_max",  8.0))
        self.goal_y_min = float(goal_range.get("y_min", -8.0))
        self.goal_y_max = float(goal_range.get("y_max",  8.0))
        if self.goal_x_min > self.goal_x_max or self.goal_y_min > self.goal_y_max:
            raise ValueError("Invalid goal_range: min must be <= max")
        if self.goal_mode == "fixed" and (
            self.goal_x_min != self.goal_x_max or self.goal_y_min != self.goal_y_max
        ):
            raise ValueError("goal_mode=fixed requires goal_range to specify a single point")

        # Curriculum: expand range by delta per reset, up to max_span beyond base
        curriculum_cfg = env_cfg.get("curriculum", {})
        self.curriculum_delta = float(curriculum_cfg.get("delta", 0.0))
        self.curriculum_max_span = float(curriculum_cfg.get("max_span", 0.0))
        self.curriculum_expansion = 0.0  # accumulated expansion so far

        # Map subscription for spawn/goal validity checks (use raw map instead of inflated costmap)
        self.map_topic = env_cfg.get("map_topic", "/map")
        self.map_grid = None
        self.map_lock = threading.Lock()
        self.obstacle_threshold = int(env_cfg.get("map_obstacle_threshold", 50))
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_cb, queue_size=1)

        # Optional opponent (e.g., robot2) control
        opponent_cfg = env_cfg.get("opponent", {})
        self.opponent_enabled = opponent_cfg.get("enabled", False)
        self.opponent_name = opponent_cfg.get("model_name", "robot2")
        self.opponent_mode = opponent_cfg.get("mode", "movebase")  # "movebase", "diffusion", or "rule_based"
        self.opponent_goal_topic = opponent_cfg.get("goal_topic", f"/{self.opponent_name}/move_base_simple/goal")
        self.opponent_frame_id = opponent_cfg.get("frame_id", "map")
        self.opponent_goal_offset = env_cfg.get("goal_offset", opponent_cfg.get("goal_offset", 0.5))
        self.opponent_spawn_radius = env_cfg.get("adv_spawn_radius", opponent_cfg.get("adv_spawn_radius", opponent_cfg.get("spawn_radius", [1.0, 3.0])))
        self.opponent_goal_pub = None
        self.opponent_cmd_pub = None
        self.rule_adversary = None
        if self.opponent_enabled:
            self.opponent_cmd_pub = rospy.Publisher(f"/{self.opponent_name}/cmd_vel", Twist, queue_size=1)
            if self.opponent_mode in ("movebase", "diffusion"):
                # Both modes use move_base: publish goal to move_base_simple/goal
                # diffusion mode uses NoMaD as global planner registered in move_base
                self.opponent_goal_pub = rospy.Publisher(self.opponent_goal_topic, PoseStamped, queue_size=1)
            elif self.opponent_mode == "rule_based":
                rule_cfg = opponent_cfg.get("rule_based", {})
                ego_name = env_cfg.get("agent_names", ["robot1"])[0]
                self.rule_adversary = RuleBasedAdversary(self.opponent_name, ego_name, rule_cfg)

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
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    def get_observations(self):
        for i, robot in enumerate(self.robots):
            obs = robot.get_observation()
            self.obs_buf[i] = torch.tensor(obs, device=self.device)
        return TensorDict({"policy": self.obs_buf, "privileged": self.obs_buf.clone()}, batch_size=[self.num_envs])

    def step(self, actions):
        actions_np = actions.detach().cpu().numpy()
        for i, robot in enumerate(self.robots):
            robot.set_action(actions_np[i])

        # Rule-based adversary publishes cmd_vel before physics step
        if self.rule_adversary is not None:
            self.rule_adversary.step()

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
            
            # Check if episode exceeded max_episode_length (convert tensor to int for comparison)
            ep_len = int(self.episode_length_buf[i].item())
            if ep_len >= self.max_episode_length:
                done = True
                if self.debug:
                    rospy.loginfo(f"[env] Episode {i} timed out at {ep_len} steps")
            
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

    def set_fixed_episode_setup(self, setup):
        self.fixed_episode_setup = dict(setup) if setup else None

    def clear_fixed_episode_setup(self):
        self.fixed_episode_setup = None

    def reset_robot(self, idx):
        robot = self.robots[idx]

        setup = self.fixed_episode_setup if idx == 0 else None
        if setup is not None:
            gx = float(setup.get("goal_x", self.goal_x_min))
            gy = float(setup.get("goal_y", self.goal_y_min))
            init_x = float(setup.get("ego_x"))
            init_y = float(setup.get("ego_y"))
            init_yaw = float(setup.get("ego_yaw", np.random.uniform(-math.pi, math.pi)))
        else:
            gx, gy = self._sample_goal()
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

        robot.init_pos = [init_x, init_y, init_yaw]
        gyaw = float(setup.get("goal_yaw")) if setup is not None and setup.get("goal_yaw") is not None else math.atan2(gy - init_y, gx - init_x)
        robot.set_absolute_goal(
            gx,
            gy,
            gyaw,
            publish_move_base=(self.robot1_mode == "movebase"),
        )
        self.last_reset_info = {
            'goal_x': float(gx),
            'goal_y': float(gy),
            'goal_yaw': float(gyaw),
            'ego_x': float(init_x),
            'ego_y': float(init_y),
            'ego_yaw': float(init_yaw),
        }

        if self.opponent_enabled and robot.model_name == self.agent_names[0]:
            self.reset_opponent(gx, gy)
            if self.opponent_mode in ("movebase", "diffusion"):
                self.publish_opponent_goal(gx, gy)
            elif self.rule_adversary is not None:
                self.rule_adversary.on_reset()

    def _sample_goal(self, max_attempts=100):
        """Sample goal from a fixed point or a curriculum-expanded rectangle."""
        if self.goal_mode == "fixed":
            return self.goal_x_min, self.goal_y_min

        if self.curriculum_delta > 0 and self.curriculum_expansion < self.curriculum_max_span:
            self.curriculum_expansion = min(
                self.curriculum_expansion + self.curriculum_delta,
                self.curriculum_max_span,
            )
            if self.debug:
                rospy.loginfo_throttle(10.0, f"[curriculum] expansion={self.curriculum_expansion:.3f}")

        x_min = self.goal_x_min - self.curriculum_expansion
        x_max = self.goal_x_max + self.curriculum_expansion
        y_min = self.goal_y_min - self.curriculum_expansion
        y_max = self.goal_y_max + self.curriculum_expansion

        for _ in range(max_attempts):
            gx = np.random.uniform(x_min, x_max)
            gy = np.random.uniform(y_min, y_max)
            if self._check_pos(gx, gy):
                return gx, gy
        rospy.logwarn("[env] _sample_goal: could not find free goal, using last sample")
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

        setup = self.fixed_episode_setup
        if setup is not None and setup.get("adv_x") is not None and setup.get("adv_y") is not None:
            ox = float(setup.get("adv_x"))
            oy = float(setup.get("adv_y"))
            oyaw = float(setup.get("adv_yaw", np.random.uniform(-math.pi, math.pi)))
        else:
            # Sample opponent position around goal, keeping >=2m from current ego pose
            r1_x, r1_y, _ = self.robots[0].init_pos if self.robots else (0.0, 0.0, 0.0)
            for attempt in range(50):
                ox, oy = self._sample_around(goal_x, goal_y, self.opponent_spawn_radius)
                if math.sqrt((ox - r1_x) ** 2 + (oy - r1_y) ** 2) >= 2.0:
                    break
                if attempt == 49:
                    rospy.logwarn("[env] Could not place opponent >=2m from robot1, using last sample")
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
        self.last_reset_info.update({
            'adv_x': float(ox),
            'adv_y': float(oy),
            'adv_yaw': float(oyaw),
        })
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

    def publish_global_goal_marker(self, goal_x, goal_y, goal_yaw):
        """Publish a global goal marker for all robots (visualization only)."""
        for robot in self.robots:
            robot.publish_goal_marker(goal_x, goal_y, goal_yaw)

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
        self.lock = threading.Lock()

        self.scan_sub = rospy.Subscriber(f"/{self.ns}/scan", LaserScan, self.scan_cb)
        self.odom_sub = rospy.Subscriber(f"/{self.ns}/odom", Odometry, self.odom_cb)
        self.cmd_pub = rospy.Publisher(f"/{self.ns}/cmd_vel", Twist, queue_size=1)
        self.goal_pub = rospy.Publisher(f"/{self.ns}/move_base_simple/goal", PoseStamped, queue_size=1)
        self.goal_marker_pub = rospy.Publisher(f"/{self.ns}/rl_goal_marker", Marker, queue_size=3)
        self.rl_goal_pub = rospy.Publisher(f"/{self.ns}/rl_goal", PoseStamped, queue_size=1)

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
            
            # Update past_distance for next step (important for progress tracking)
            if self.past_distance is None:
                self.past_distance = dist_to_goal
            else:
                self.past_distance = dist_to_goal

        if self.debug:
            rospy.loginfo(
                f"dist={dist_to_goal:.2f} min_scan={min_scan:.2f}\n "
                f"cmd=({action[0]:.2f},{action[1]:.2f})\n "
                f"reward={reward:.2f}\n"
                f"R_v={action[0] / 2} R_w={- abs(action[1]) / 2}"
            )

        
        return float(reward), done

    def set_absolute_goal(self, goal_x, goal_y, goal_yaw, publish_move_base=False):
        """Set goal state and publish for visualization.

        Args:
            goal_x, goal_y, goal_yaw: Target goal position and orientation
            publish_move_base: If True, also publish to move_base_simple/goal for move_base navigation
        """
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_yaw = goal_yaw
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.past_distance = None

        # Build goal pose
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

        # Publish to rl_goal topic for visualization
        self.rl_goal_pub.publish(goal)

        # Optionally publish to move_base_simple/goal for move_base navigation
        if publish_move_base:
            self.goal_pub.publish(goal)

        self.publish_markers(np.zeros(2, dtype=np.float32))

    def _get_yaw(self, q):
        orientation_list = [q.x, q.y, q.z, q.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation_list)
        return yaw

    def publish_goal_marker(self, goal_x, goal_y, goal_yaw):
        """Publish a goal marker at the specified position (for global goal visualization)."""
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.ns = f"/{self.ns}/global_goal"
        goal_marker.id = 0
        goal_marker.type = goal_marker.CYLINDER
        goal_marker.action = goal_marker.ADD
        goal_marker.scale.x = 0.3
        goal_marker.scale.y = 0.3
        goal_marker.scale.z = 0.3
        goal_marker.color.a = 1.0
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.pose.position.x = goal_x
        goal_marker.pose.position.y = goal_y
        goal_marker.pose.position.z = 0.0

        self.goal_marker_pub.publish(goal_marker)

    def publish_markers(self, action):
        # Publish goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.ns = f"/{self.ns}/rl_goal"
        goal_marker.id = 1
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
