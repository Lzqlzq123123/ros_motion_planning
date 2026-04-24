import os
import argparse
import csv
import json
import yaml
import numpy as np
import torch
import time
import sys
import os.path as osp
import threading
import random

import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import tf.transformations
import math

proj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(proj_root, 'src'))

from rl_training.third_party.drl_td3.td3_models import TD3
from rl_training.envs.movebase_gazebo_env import MoveBaseGazeboEnv


class EpisodeImageRecorder:
    def __init__(
        self,
        enabled=False,
        image_topic=None,
        output_root='/tmp/eval_images',
        save_rate=5.0,
        record_all_episodes=False,
        selected_episode_indices=None,
        enable_rviz_shot=False,
        save_rate_rviz=0.0,
        rviz_window_title="rviz",
        rviz_restore_before_shot=True,
        rviz_hide_after_shot=False,
    ):
        self.enabled = enabled
        self.image_topic = image_topic
        self.output_root = output_root
        self.save_rate = float(save_rate)
        self.record_all_episodes = bool(record_all_episodes)
        self.selected_episode_indices = set(selected_episode_indices or [])
        
        self.enable_rviz_shot = enable_rviz_shot
        self.save_rate_rviz = float(save_rate_rviz)
        self.rviz_window_title = rviz_window_title
        self.rviz_restore_before_shot = rviz_restore_before_shot
        self.rviz_hide_after_shot = rviz_hide_after_shot
        
        self.lock = threading.Lock()
        self.recording = False
        self.episode_dir = None
        self.frame_count = 0
        self.rviz_frame_count = 0
        self.last_save_time = rospy.Time(0)
        self.last_rviz_time = 0.0
        self.episode_metadata = None
        self.sub = None
        self.screenshot_srv = None

        if self.enabled or self.enable_rviz_shot:
            os.makedirs(self.output_root, exist_ok=True)
            if self.enabled and self.image_topic:
                self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
                rospy.loginfo('EpisodeImageRecorder subscribed to %s', self.image_topic)
            if self.enable_rviz_shot:
                rospy.loginfo("Waiting for /rviz/screenshot service ...")
                try:
                    rospy.wait_for_service("/rviz/screenshot", timeout=2.0)
                    rospy.loginfo("Connected to /rviz/screenshot service")
                    try:
                        from jsk_rviz_plugins.srv import Screenshot
                        self.screenshot_srv = rospy.ServiceProxy("/rviz/screenshot", Screenshot)
                        rospy.loginfo("Using jsk_rviz_plugins.srv.Screenshot (Fast Native RPC)")
                    except ImportError:
                        rospy.logwarn("jsk_rviz_plugins not found! Will fallback to slow 'rosservice call'.")
                except rospy.ROSException:
                    rospy.logwarn("Could not connect to /rviz/screenshot service within 2s.")
                
                self.rviz_thread = threading.Thread(target=self._rviz_shot_loop)
                self.rviz_thread.daemon = True
                self.rviz_thread.start()

    def _has_wmctrl(self):
        try:
            import subprocess
            subprocess.run(["wmctrl", "-m"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def _wmctrl_restore(self):
        import subprocess
        subprocess.run(["wmctrl", "-R", self.rviz_window_title],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wmctrl_hide(self):
        import subprocess
        try:
            p = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=True)
            lines = p.stdout.splitlines()
            win_id = None
            for line in lines:
                if self.rviz_window_title.lower() in line.lower():
                    win_id = line.split()[0]
                    break
            if win_id:
                subprocess.run(["wmctrl", "-i", "-r", win_id, "-b", "add,hidden"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _rviz_shot_loop(self):
        target_rate = self.save_rate_rviz if self.save_rate_rviz > 0 else 2.0
        
        while not rospy.is_shutdown():
            if not self.enable_rviz_shot or not self.recording or self.episode_dir is None:
                time.sleep(0.1)
                continue
                
            now = time.time()
            if now - self.last_rviz_time < 1.0 / target_rate:
                time.sleep(0.01)
                continue
            
            self.last_rviz_time = now
            filename = os.path.join(self.episode_dir, f"rviz_{self.rviz_frame_count:06d}.png")

            if self.rviz_restore_before_shot and self._has_wmctrl():
                self._wmctrl_restore()

            try:
                if self.screenshot_srv is not None:
                    self.screenshot_srv(filename)
                else:
                    import subprocess
                    subprocess.run(["rosservice", "call", "/rviz/screenshot", filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.rviz_frame_count += 1
            except Exception as e:
                rospy.logwarn_throttle(2.0, "RViz screenshot failed: %s", str(e))

            if self.rviz_hide_after_shot and self._has_wmctrl():
                self._wmctrl_hide()

    def update(self):
        pass

    def should_record_episode(self, episode_idx):
        if not (self.enabled or self.enable_rviz_shot):
            return False
        if self.record_all_episodes:
            return True
        return episode_idx in self.selected_episode_indices

    def start_episode(self, episode_idx, metadata=None):
        if not (self.enabled or self.enable_rviz_shot):
            return
        with self.lock:
            self.episode_dir = os.path.join(self.output_root, f'episode_{episode_idx + 1:03d}')
            os.makedirs(self.episode_dir, exist_ok=True)
            self.frame_count = 0
            self.rviz_frame_count = 0
            self.last_save_time = rospy.Time(0)
            self.last_rviz_time = 0.0
            self.episode_metadata = dict(metadata or {})
            self.episode_metadata.update({
                'episode_index_0based': int(episode_idx),
                'episode_index_1based': int(episode_idx + 1),
                'image_topic': self.image_topic if self.enabled else None,
                'save_rate_hz': self.save_rate,
                'enable_rviz_shot': self.enable_rviz_shot,
                'save_rate_rviz': self.save_rate_rviz,
            })
            self._write_metadata_locked()
            self.recording = True
            rospy.loginfo('Recording episode images to %s', self.episode_dir)

    def stop_episode(self, metadata=None):
        if not (self.enabled or self.enable_rviz_shot):
            return
        with self.lock:
            if metadata:
                if self.episode_metadata is None:
                    self.episode_metadata = {}
                self.episode_metadata.update(metadata)
            if self.episode_metadata is None:
                self.episode_metadata = {}
            self.episode_metadata['saved_frame_count'] = int(self.frame_count)
            self.episode_metadata['saved_rviz_frame_count'] = int(self.rviz_frame_count)
            self._write_metadata_locked()
            self.recording = False

    def _write_metadata_locked(self):
        if self.episode_dir is None or self.episode_metadata is None:
            return
        meta_path = os.path.join(self.episode_dir, 'episode_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.episode_metadata, f, ensure_ascii=False, indent=2)

    def image_cb(self, msg):
        with self.lock:
            if not self.recording or self.episode_dir is None:
                return

            now = rospy.Time.now()
            if self.save_rate > 0 and (now - self.last_save_time).to_sec() < 1.0 / self.save_rate:
                return

            if msg.encoding not in ['rgb8', 'bgr8']:
                rospy.logwarn_throttle(5.0, 'Unsupported image encoding: %s', msg.encoding)
                return

            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'bgr8':
                image = image[:, :, ::-1]

            image_path = os.path.join(self.episode_dir, f'frame_{self.frame_count:06d}.png')
            PILImage.fromarray(image).save(image_path)
            self.frame_count += 1
            self.last_save_time = now


def parse_episode_indices(spec):
    indices = set()
    if not spec:
        return indices
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_s, end_s = chunk.split('-', 1)
            start_i = int(start_s)
            end_i = int(end_s)
            if start_i <= 0 or end_i <= 0:
                raise ValueError('episode indices must be positive integers')
            if end_i < start_i:
                raise ValueError('episode range end must be >= start')
            for idx in range(start_i, end_i + 1):
                indices.add(idx - 1)
        else:
            idx = int(chunk)
            if idx <= 0:
                raise ValueError('episode indices must be positive integers')
            indices.add(idx - 1)
    return indices


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_goal(args):
    if args.goal_x is None or args.goal_y is None:
        return None
    goal_yaw = args.goal_yaw
    if goal_yaw is None:
        goal_yaw = float(np.arctan2(args.goal_y, args.goal_x))
    return [float(args.goal_x), float(args.goal_y), float(goal_yaw)]


def apply_eval_overrides(env_cfg, args):
    env_cfg = dict(env_cfg)
    env_cfg['robot1_mode'] = args.robot1_mode

    if args.max_steps is not None:
        env_cfg['max_episode_length'] = int(args.max_steps)

    if any(v is not None for v in [args.ego_spawn_radius_min, args.ego_spawn_radius_max]):
        if None in [args.ego_spawn_radius_min, args.ego_spawn_radius_max]:
            raise ValueError('ego_spawn_radius override requires both min and max')
        if args.ego_spawn_radius_min > args.ego_spawn_radius_max:
            raise ValueError('ego_spawn_radius_min must be <= ego_spawn_radius_max')
        env_cfg['ego_spawn_radius'] = [
            float(args.ego_spawn_radius_min),
            float(args.ego_spawn_radius_max),
        ]

    if any(v is not None for v in [args.adv_spawn_radius_min, args.adv_spawn_radius_max]):
        if None in [args.adv_spawn_radius_min, args.adv_spawn_radius_max]:
            raise ValueError('adv_spawn_radius override requires both min and max')
        if args.adv_spawn_radius_min > args.adv_spawn_radius_max:
            raise ValueError('adv_spawn_radius_min must be <= adv_spawn_radius_max')
        env_cfg['adv_spawn_radius'] = [
            float(args.adv_spawn_radius_min),
            float(args.adv_spawn_radius_max),
        ]
        opponent_cfg = dict(env_cfg.get('opponent', {}))
        if opponent_cfg:
            opponent_cfg['adv_spawn_radius'] = [
                float(args.adv_spawn_radius_min),
                float(args.adv_spawn_radius_max),
            ]
            env_cfg['opponent'] = opponent_cfg

    if any(v is not None for v in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]):
        if None in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]:
            raise ValueError('goal_range override requires goal_x_min, goal_x_max, goal_y_min, goal_y_max')
        if args.goal_x_min > args.goal_x_max:
            raise ValueError('goal_x_min must be <= goal_x_max')
        if args.goal_y_min > args.goal_y_max:
            raise ValueError('goal_y_min must be <= goal_y_max')
        env_cfg['goal_range'] = {
            'x_min': float(args.goal_x_min),
            'x_max': float(args.goal_x_max),
            'y_min': float(args.goal_y_min),
            'y_max': float(args.goal_y_max),
        }
        env_cfg['goal_mode'] = 'random'

    if args.goal_x is not None and args.goal_y is not None:
        env_cfg['goal_range'] = {
            'x_min': float(args.goal_x),
            'x_max': float(args.goal_x),
            'y_min': float(args.goal_y),
            'y_max': float(args.goal_y),
        }
        env_cfg['goal_mode'] = 'fixed'

    opponent_cfg = dict(env_cfg.get('opponent', {}))
    if args.opponent_mode is not None:
        opponent_cfg['mode'] = args.opponent_mode
    if opponent_cfg:
        env_cfg['opponent'] = opponent_cfg

    if args.opponent_goal_offset is not None:
        env_cfg['goal_offset'] = float(args.opponent_goal_offset)

    return env_cfg


def append_episode_result(csv_path, row):
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    fieldnames = [
        'row_type',
        'episode',
        'robot1_mode',
        'opponent_mode',
        'opponent_enabled',
        'collided',
        'reached_goal',
        'timed_out',
        'steps',
        'episode_reward',
        'goal_x',
        'goal_y',
        'goal_yaw',
        'model_path',
        'avg_reward',
        'success_rate',
        'collision_rate',
        'timeout_rate',
        'min_ttc',
        'frechet_distance',
        'smoothness',
    ]

    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_summary_result(csv_path, row):
    append_episode_result(csv_path, row)


def save_episode_trajectory(output_root, episode_idx, ego_traj, adv_traj, metadata=None):
    if not output_root:
        return
    episode_dir = os.path.join(output_root, f'episode_{episode_idx + 1:03d}')
    os.makedirs(episode_dir, exist_ok=True)

    traj_path = os.path.join(episode_dir, 'trajectory.csv')
    with open(traj_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'ego_x', 'ego_y', 'adv_x', 'adv_y'])
        num_steps = min(len(ego_traj), len(adv_traj))
        for step in range(num_steps):
            writer.writerow([
                step,
                float(ego_traj[step][0]),
                float(ego_traj[step][1]),
                float(adv_traj[step][0]),
                float(adv_traj[step][1]),
            ])

    if metadata is None:
        metadata = {}
    meta_path = os.path.join(episode_dir, 'trajectory_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


class OdomTracker:
    def __init__(self, topic):
        self.odom = None
        self.sub = rospy.Subscriber(topic, Odometry, self.cb)

    def cb(self, msg):
        self.odom = msg

def discrete_frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    if n == 0 or m == 0:
        return 0.0
    ca = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d = math.hypot(P[i][0] - Q[j][0], P[i][1] - Q[j][1])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i-1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j-1], d)
            else:
                ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)
    return ca[n-1, m-1]

def evaluate(
    network,
    env,
    device,
    eval_episodes=10,
    step_limit=5000,
    recorder=None,
    csv_path=None,
    run_info=None,
    trajectory_output_root=None,
    record_all_trajectory_episodes=False,
    selected_trajectory_episode_indices=None,
    episode_setups=None,
    save_episode_setups_json=None,
):
    avg_reward = 0.0
    collision_episodes = 0
    goal_episodes = 0
    timeout_episodes = 0
    valid_episodes = 0  # Count of episodes with step_count >= 10

    total_min_ttc = 0.0
    total_frechet = 0.0
    total_smoothness = 0.0

    ego_tracker = OdomTracker('/robot1/odom')
    adv_tracker = OdomTracker('/robot2/odom')
    used_episode_setups = []

    print(f"Starting evaluation over {eval_episodes} episodes...")

    for episode_idx in range(eval_episodes):
        fixed_setup = None
        if episode_setups is not None and episode_idx < len(episode_setups):
            fixed_setup = dict(episode_setups[episode_idx])
        if hasattr(env, 'set_fixed_episode_setup'):
            env.set_fixed_episode_setup(fixed_setup)
        obs_td = env.reset()
        reset_info = dict(getattr(env, 'last_reset_info', {}) or {})
        used_episode_setups.append(reset_info)
        episode_goal = [env.robots[0].goal_x, env.robots[0].goal_y, env.robots[0].goal_yaw]

        should_record = recorder is not None and recorder.should_record_episode(episode_idx)
        if should_record:
            recorder.start_episode(
                episode_idx,
                metadata={
                    'goal_x': float(episode_goal[0]) if episode_goal is not None else None,
                    'goal_y': float(episode_goal[1]) if episode_goal is not None else None,
                    'goal_yaw': float(episode_goal[2]) if episode_goal is not None else None,
                    'ego_x': reset_info.get('ego_x'),
                    'ego_y': reset_info.get('ego_y'),
                    'ego_yaw': reset_info.get('ego_yaw'),
                    'adv_x': reset_info.get('adv_x'),
                    'adv_y': reset_info.get('adv_y'),
                    'adv_yaw': reset_info.get('adv_yaw'),
                    'robot1_mode': run_info.get('robot1_mode') if run_info else None,
                    'opponent_mode': run_info.get('opponent_mode') if run_info else None,
                    'opponent_enabled': bool(run_info.get('opponent_enabled')) if run_info else None,
                    'model_path': run_info.get('model_path') if run_info else None,
                },
            )

        state = obs_td['policy'][0].cpu().numpy()
        done = False
        step_count = 0
        episode_reward = 0.0
        collided = False
        reached_goal = False

        episode_ttc_min = float('inf')
        ego_traj = []
        adv_traj = []

        while not done and step_count < step_limit:
            if getattr(env, 'robot1_mode', 'td3') == 'movebase':
                a_in = np.array([0.0, 0.0])
            else:
                action = network.get_action(np.array(state))
                a_in = np.array([(action[0] + 1) / 2.0, action[1]])

            action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
            next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)

            state = next_obs_td['policy'][0].cpu().numpy()
            reward = float(reward_tensor[0].cpu().item())
            done = bool(done_tensor[0].cpu().item())
            episode_reward += reward
            step_count += 1

            # Record trajectory and TTC
            ego_odom = ego_tracker.odom
            adv_odom = adv_tracker.odom
            if ego_odom is not None and adv_odom is not None:
                p_ego = np.array([ego_odom.pose.pose.position.x, ego_odom.pose.pose.position.y])
                v_ego = np.array([ego_odom.twist.twist.linear.x, ego_odom.twist.twist.linear.y])
                p_adv = np.array([adv_odom.pose.pose.position.x, adv_odom.pose.pose.position.y])
                v_adv = np.array([adv_odom.twist.twist.linear.x, adv_odom.twist.twist.linear.y])

                ego_traj.append(p_ego)
                adv_traj.append(p_adv)

                rel_v = np.linalg.norm(v_adv - v_ego)
                if rel_v > 0.01:
                    ttc = np.linalg.norm(p_adv - p_ego) / rel_v
                    if ttc < episode_ttc_min:
                        episode_ttc_min = ttc

            if should_record and recorder is not None:
                recorder.update()

            if reward < -90:
                collided = True
            elif reward > 90:
                reached_goal = True

        timed_out = (not done and step_count >= step_limit)
        if should_record:
            recorder.stop_episode(
                metadata={
                    'collided': bool(collided),
                    'reached_goal': bool(reached_goal),
                    'timed_out': bool(timed_out),
                    'steps': int(step_count),
                    'episode_reward': float(episode_reward),
                    'min_ttc': None if episode_ttc_min == float('inf') else float(episode_ttc_min),
                }
            )

        should_save_traj = bool(trajectory_output_root) and (
            record_all_trajectory_episodes or episode_idx in (selected_trajectory_episode_indices or set())
        )

        # Skip episodes with step_count < 10
        if step_count < 10:
            if should_save_traj and len(ego_traj) > 0 and len(adv_traj) > 0:
                save_episode_trajectory(
                    trajectory_output_root,
                    episode_idx,
                    ego_traj,
                    adv_traj,
                    metadata={
                        'robot1_mode': run_info.get('robot1_mode') if run_info else None,
                        'opponent_mode': run_info.get('opponent_mode') if run_info else None,
                        'collided': bool(collided),
                        'reached_goal': bool(reached_goal),
                        'timed_out': bool(timed_out),
                        'steps': int(step_count),
                        'skipped': True,
                    },
                )
            print(
                f"Episode {episode_idx + 1}: reward={episode_reward:.2f}, "
                f"steps={step_count}, collided={collided}, reached_goal={reached_goal} [SKIPPED - steps < 10]"
            )
            continue

        frechet_dist = discrete_frechet_distance(adv_traj, ego_traj)

        smoothness = 0.0
        if len(adv_traj) > 2:
            adv_traj_np = np.array(adv_traj)
            diff2 = adv_traj_np[2:] - 2*adv_traj_np[1:-1] + adv_traj_np[:-2]
            smoothness = np.sum(np.linalg.norm(diff2, axis=1)**2) / (len(adv_traj) - 2)

        # Only increment counters and accumulate metrics for valid episodes (step_count >= 10)
        valid_episodes += 1
        total_min_ttc += episode_ttc_min if episode_ttc_min != float('inf') else 10.0
        total_frechet += frechet_dist
        total_smoothness += smoothness

        # Each episode belongs to exactly one category (mutually exclusive)
        if collided:
            collision_episodes += 1
        elif reached_goal:
            goal_episodes += 1
        elif not done and step_count >= step_limit:
            timeout_episodes += 1

        if csv_path is not None:
            goal_x, goal_y, goal_yaw = (episode_goal if episode_goal is not None else [None, None, None])
            append_episode_result(
                csv_path,
                {
                    'episode': episode_idx + 1,
                    'robot1_mode': run_info.get('robot1_mode') if run_info else None,
                    'opponent_mode': run_info.get('opponent_mode') if run_info else None,
                    'opponent_enabled': run_info.get('opponent_enabled') if run_info else None,
                    'collided': int(collided),
                    'reached_goal': int(reached_goal),
                    'timed_out': int(timed_out),
                    'steps': step_count,
                    'episode_reward': episode_reward,
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                    'goal_yaw': goal_yaw,
                    'model_path': run_info.get('model_path') if run_info else None,
                    'min_ttc': episode_ttc_min if episode_ttc_min != float('inf') else None,
                    'frechet_distance': frechet_dist,
                    'smoothness': smoothness,
                },
            )

        if should_save_traj and len(ego_traj) > 0 and len(adv_traj) > 0:
            save_episode_trajectory(
                trajectory_output_root,
                episode_idx,
                ego_traj,
                adv_traj,
                metadata={
                    'robot1_mode': run_info.get('robot1_mode') if run_info else None,
                    'opponent_mode': run_info.get('opponent_mode') if run_info else None,
                    'collided': bool(collided),
                    'reached_goal': bool(reached_goal),
                    'timed_out': bool(timed_out),
                    'steps': int(step_count),
                    'episode_reward': float(episode_reward),
                    'min_ttc': episode_ttc_min if episode_ttc_min != float('inf') else None,
                    'frechet_distance': float(frechet_dist),
                    'smoothness': float(smoothness),
                    'goal_x': float(episode_goal[0]) if episode_goal is not None else None,
                    'goal_y': float(episode_goal[1]) if episode_goal is not None else None,
                    'goal_yaw': float(episode_goal[2]) if episode_goal is not None else None,
                    'ego_x': reset_info.get('ego_x'),
                    'ego_y': reset_info.get('ego_y'),
                    'ego_yaw': reset_info.get('ego_yaw'),
                    'adv_x': reset_info.get('adv_x'),
                    'adv_y': reset_info.get('adv_y'),
                    'adv_yaw': reset_info.get('adv_yaw'),
                    'skipped': False,
                },
            )

        avg_reward += episode_reward
        print(
            f"Episode {episode_idx + 1}: reward={episode_reward:.2f}, "
            f"steps={step_count}, collided={collided}, reached_goal={reached_goal}, "
            f"min_ttc={episode_ttc_min if episode_ttc_min != float('inf') else 'inf'}, frechet={frechet_dist:.2f}, smooth={smoothness:.4f}"
        )

    # Calculate rates based on valid episodes only
    avg_reward = avg_reward / valid_episodes if valid_episodes > 0 else 0.0
    collision_rate = collision_episodes / valid_episodes if valid_episodes > 0 else 0.0
    success_rate = goal_episodes / valid_episodes if valid_episodes > 0 else 0.0
    timeout_rate = timeout_episodes / valid_episodes if valid_episodes > 0 else 0.0
    avg_min_ttc = total_min_ttc / valid_episodes if valid_episodes > 0 else 0.0
    avg_frechet = total_frechet / valid_episodes if valid_episodes > 0 else 0.0
    avg_smoothness = total_smoothness / valid_episodes if valid_episodes > 0 else 0.0

    print("..............................................")
    print(f"Evaluation Results (Valid Episodes: {valid_episodes}/{eval_episodes}):")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Collision Rate: {collision_rate:.4f}")
    print(f"Timeout Rate: {timeout_rate:.4f}")
    print(f"Average Min TTC: {avg_min_ttc:.4f}")
    print(f"Average Frechet Dist: {avg_frechet:.4f}")
    print(f"Average Smoothness: {avg_smoothness:.4f}")
    print("..............................................")

    if save_episode_setups_json is not None:
        with open(save_episode_setups_json, 'w', encoding='utf-8') as f:
            json.dump(used_episode_setups, f, ensure_ascii=False, indent=2)

    if csv_path is not None:
        append_summary_result(
            csv_path,
            {
                'row_type': 'summary',
                'episode': 'all',
                'robot1_mode': run_info.get('robot1_mode') if run_info else None,
                'opponent_mode': run_info.get('opponent_mode') if run_info else None,
                'opponent_enabled': run_info.get('opponent_enabled') if run_info else None,
                'model_path': run_info.get('model_path') if run_info else None,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'timeout_rate': timeout_rate,
                'min_ttc': avg_min_ttc,
                'frechet_distance': avg_frechet,
                'smoothness': avg_smoothness,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/forklift_movebase.yaml")
    parser.add_argument('--model_path', type=str, default=None, help='Path to model (without _actor.pth); required when --robot1_mode td3')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=None, help='Override max_episode_length from yaml')
    # Fixed goal: if provided, every episode uses this exact goal
    parser.add_argument('--goal_x', type=float, default=None)
    parser.add_argument('--goal_y', type=float, default=None)
    parser.add_argument('--goal_yaw', type=float, default=None)
    # Goal range overrides: override yaml goal_range for random sampling
    parser.add_argument('--goal_x_min', type=float, default=None)
    parser.add_argument('--goal_x_max', type=float, default=None)
    parser.add_argument('--goal_y_min', type=float, default=None)
    parser.add_argument('--goal_y_max', type=float, default=None)
    parser.add_argument('--ego_spawn_radius_min', type=float, default=None)
    parser.add_argument('--ego_spawn_radius_max', type=float, default=None)
    parser.add_argument('--adv_spawn_radius_min', type=float, default=None)
    parser.add_argument('--adv_spawn_radius_max', type=float, default=None)
    parser.add_argument('--robot1_mode', type=str, choices=['td3', 'movebase'], default='td3')
    parser.add_argument('--opponent_mode', type=str, choices=['movebase', 'diffusion', 'rule_based'], default=None)
    parser.add_argument('--opponent_goal_offset', type=float, default=None)
    
    # Image recording args
    parser.add_argument('--record_image_topic', type=str, default=None, help='Generic sensor_msgs/Image topic to record per episode, e.g. /robot2/camera/rgb/image_raw')
    parser.add_argument('--record_image_output_dir', type=str, default='/tmp/eval_images', help='Output directory for recorded per-episode images')
    parser.add_argument('--record_image_save_rate', type=float, default=5, help='Frame save rate for --record_image_topic; 0 means save every frame')
    
    # RViz recording args
    parser.add_argument('--enable_rviz_shot', action='store_true', help='Enable screenshotting RViz per episode')
    parser.add_argument('--save_rate_rviz', type=float, default=100.0, help='Frame save rate for RViz screenshots')
    parser.add_argument('--rviz_window_title', type=str, default='rviz', help='Window title for RViz (wmctrl matching)')

    parser.add_argument('--record_all_image_episodes', action='store_true', help='Record all evaluation episodes for the selected image topic')
    parser.add_argument('--record_episode_indices', type=str, default=None, help='1-based episode list/range, e.g. 2,5-7')
    parser.add_argument('--record_trajectory_output_dir', type=str, default=None, help='Optional output root to save per-episode ego/adv trajectories')
    parser.add_argument('--record_all_trajectory_episodes', action='store_true', help='Save trajectories for all episodes')
    parser.add_argument('--record_trajectory_episode_indices', type=str, default=None, help='1-based trajectory episode list/range, e.g. 1,3-5')
    parser.add_argument('--episode_setup_json', type=str, default=None, help='Optional JSON list of fixed per-episode setups: ego/adv start pose + goal')
    parser.add_argument('--save_episode_setups_json', type=str, default=None, help='Optional output JSON path to save actual per-episode reset setups')
    parser.add_argument('--random_seed', type=int, default=None, help='Seed numpy/random/torch for reproducible episode sampling')
    parser.add_argument('--csv_path', type=str, default='/data/lzq/ros_motion_planning/src/rl_training/eval_results.csv')
    args = parser.parse_args()

    if args.goal_x is not None and args.goal_y is None:
        print('Invalid goal configuration: both goal_x and goal_y are required for fixed goal')
        sys.exit(2)
    if args.goal_y is not None and args.goal_x is None:
        print('Invalid goal configuration: both goal_x and goal_y are required for fixed goal')
        sys.exit(2)
    if args.goal_x is not None and any(v is not None for v in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]):
        print('Invalid goal configuration: choose either fixed goal or goal_range override')
        sys.exit(2)

    cfg = load_yaml(args.config)
    try:
        env_cfg = apply_eval_overrides(cfg['env'], args)
    except ValueError as exc:
        print(f"Invalid goal configuration: {exc}")
        sys.exit(2)

    # Set max_steps from yaml if not provided via CLI
    max_steps = args.max_steps if args.max_steps is not None else env_cfg.get('max_episode_length', 500)
    print(f"[eval] Using max_steps: {max_steps}")

    if args.random_seed is not None:
        np.random.seed(int(args.random_seed))
        random.seed(int(args.random_seed))
        torch.manual_seed(int(args.random_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.random_seed))
        print(f"[eval] Using random_seed: {args.random_seed}")

    device = torch.device(args.device)
    env = MoveBaseGazeboEnv(env_cfg, device=args.device)
    time.sleep(2)

    selected_episode_indices = set()
    if args.record_episode_indices:
        try:
            selected_episode_indices = parse_episode_indices(args.record_episode_indices)
        except ValueError as exc:
            print(f'Invalid --record_episode_indices: {exc}')
            sys.exit(2)
    elif (args.record_image_topic or args.enable_rviz_shot) and not args.record_all_image_episodes:
        selected_episode_indices = {0}

    record_image_enabled = bool(args.record_image_topic)

    recorder = EpisodeImageRecorder(
        enabled=record_image_enabled,
        image_topic=args.record_image_topic,
        output_root=args.record_image_output_dir,
        save_rate=args.record_image_save_rate,
        record_all_episodes=args.record_all_image_episodes,
        selected_episode_indices=selected_episode_indices,
        enable_rviz_shot=args.enable_rviz_shot,
        save_rate_rviz=args.save_rate_rviz,
        rviz_window_title=args.rviz_window_title,
        rviz_restore_before_shot=True,
        rviz_hide_after_shot=False,
    )

    selected_trajectory_episode_indices = set()
    if args.record_trajectory_episode_indices:
        try:
            selected_trajectory_episode_indices = parse_episode_indices(args.record_trajectory_episode_indices)
        except ValueError as exc:
            print(f'Invalid --record_trajectory_episode_indices: {exc}')
            sys.exit(2)
    elif args.record_trajectory_output_dir and not args.record_all_trajectory_episodes:
        selected_trajectory_episode_indices = {0}

    episode_setups = None
    if args.episode_setup_json is not None:
        with open(args.episode_setup_json, 'r', encoding='utf-8') as f:
            episode_setups = json.load(f)
        print(f"[eval] Loaded fixed episode setups: {len(episode_setups)} from {args.episode_setup_json}")

    network = None
    if args.robot1_mode == 'td3':
        if not args.model_path:
            print("--model_path is required when --robot1_mode td3")
            sys.exit(2)

        state_dim = env_cfg.get('num_observations')
        action_dim = env_cfg.get('num_actions')
        max_action = 1
        network = TD3(state_dim, action_dim, max_action, device=device)

        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path)
        print(f"Loading model: {model_name} from {model_dir}")

        try:
            network.load(model_name, model_dir)
            print("Model loaded successfully.")
        except Exception as exc:
            print(f"Failed to load model: {exc}")
            sys.exit(1)
    else:
        print("robot1_mode=movebase, skipping TD3 model loading.")

    evaluate(
        network=network,
        env=env,
        device=device,
        eval_episodes=args.episodes,
        step_limit=max_steps,
        recorder=recorder,
        csv_path=args.csv_path,
        run_info={
            'robot1_mode': args.robot1_mode,
            'opponent_mode': args.opponent_mode if args.opponent_mode is not None else env_cfg.get('opponent', {}).get('mode'),
            'opponent_enabled': bool(env_cfg.get('opponent', {}).get('enabled', False)),
            'model_path': args.model_path,
        },
        trajectory_output_root=args.record_trajectory_output_dir,
        record_all_trajectory_episodes=args.record_all_trajectory_episodes,
        selected_trajectory_episode_indices=selected_trajectory_episode_indices,
        episode_setups=episode_setups,
        save_episode_setups_json=args.save_episode_setups_json,
    )
