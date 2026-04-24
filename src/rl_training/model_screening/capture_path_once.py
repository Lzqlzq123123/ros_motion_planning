#!/usr/bin/env python3
import argparse
import csv
import os
import time

import rospy
from nav_msgs.msg import Path


class PathCapture:
    def __init__(self, topic: str, output_csv: str, timeout: float, idle_after_first: float, min_poses: int):
        self.topic = topic
        self.output_csv = output_csv
        self.timeout = float(timeout)
        self.idle_after_first = float(idle_after_first)
        self.min_poses = int(min_poses)
        self.latest_msg = None
        self.last_msg_time = None
        self.first_msg_time = None
        self.sub = rospy.Subscriber(topic, Path, self.cb, queue_size=1)

    def cb(self, msg: Path):
        if msg is None or len(msg.poses) < self.min_poses:
            return
        self.latest_msg = msg
        now = time.time()
        self.last_msg_time = now
        if self.first_msg_time is None:
            self.first_msg_time = now
        rospy.loginfo_throttle(1.0, '[capture_path_once] received path with %d poses on %s', len(msg.poses), self.topic)

    def run(self):
        start = time.time()
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                now = time.time()
                if self.latest_msg is not None:
                    if self.idle_after_first > 0 and self.last_msg_time is not None and (now - self.last_msg_time) >= self.idle_after_first:
                        break
                if (now - start) >= self.timeout:
                    break
                rate.sleep()
        except rospy.ROSInterruptException:
            rospy.logwarn('[capture_path_once] ROS shutdown before idle timeout; saving latest received path if any')

        if self.latest_msg is None:
            raise RuntimeError(f'No path received on {self.topic} within {self.timeout}s')

        os.makedirs(os.path.dirname(self.output_csv) or '.', exist_ok=True)
        with open(self.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'x', 'y'])
            for i, pose_stamped in enumerate(self.latest_msg.poses):
                writer.writerow([i, pose_stamped.pose.position.x, pose_stamped.pose.position.y])
        rospy.loginfo('[capture_path_once] saved %d poses to %s', len(self.latest_msg.poses), self.output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--idle-after-first', type=float, default=2.0)
    parser.add_argument('--min-poses', type=int, default=2)
    args = parser.parse_args()

    rospy.init_node('capture_path_once', anonymous=True)
    cap = PathCapture(args.topic, args.output_csv, args.timeout, args.idle_after_first, args.min_poses)
    cap.run()


if __name__ == '__main__':
    main()
