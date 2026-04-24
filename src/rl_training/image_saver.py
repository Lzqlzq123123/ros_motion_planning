#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Empty

def ts_str():
    t = time.time()
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + ("_%03d" % int((t * 1000) % 1000))

class RGBAndRvizScreenshot:
    def __init__(self):
        self.bridge = CvBridge()

        # 两路相机话题
        self.topic_a = rospy.get_param("~topic_a", "/robot2/camera/rgb/image_raw")
        self.topic_b = rospy.get_param("~topic_b", "/robot1/camera/rgb/image_raw")

        # 输出目录
        self.out_dir = os.path.expanduser(rospy.get_param("~out_dir", "~/ros_images"))
        self.dir_a = os.path.join(self.out_dir, "robot2_rgb")
        self.dir_b = os.path.join(self.out_dir, "robot1_rgb")
        self.dir_rviz = os.path.join(self.out_dir, "rviz")
        for d in [self.dir_a, self.dir_b, self.dir_rviz]:
            os.makedirs(d, exist_ok=True)

        # 图片保存参数
        self.ext = rospy.get_param("~ext", "jpg").lower()  # jpg 或 png
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))
        self.save_rate_img = float(rospy.get_param("~save_rate_img", 10.0))  # Hz, 0=每帧

        # RViz截图参数
        self.enable_rviz_shot = bool(rospy.get_param("~enable_rviz_shot", True))
        self.save_rate_rviz = float(rospy.get_param("~save_rate_rviz", 0))  # Hz, 0=每次都截(不建议)
        self.rviz_window_title = rospy.get_param("~rviz_window_title", "rviz")  # wmctrl 匹配窗口标题
        self.rviz_restore_before_shot = bool(rospy.get_param("~rviz_restore_before_shot", True))
        self.rviz_hide_after_shot = bool(rospy.get_param("~rviz_hide_after_shot", False))
        self.rviz_restore_delay = float(rospy.get_param("~rviz_restore_delay", 0.2))  # 秒

        # 内部限速
        self.last_a = 0.0
        self.last_b = 0.0
        self.last_rviz = 0.0

        # 订阅两路相机
        self.sub_a = rospy.Subscriber(self.topic_a, Image, self.cb_a, queue_size=1)
        self.sub_b = rospy.Subscriber(self.topic_b, Image, self.cb_b, queue_size=1)

        # RViz截图 service
        self.srv = None
        if self.enable_rviz_shot:
            rospy.loginfo("Waiting for /rviz/screenshot service ...")
            rospy.wait_for_service("/rviz/screenshot")
            self.srv = rospy.ServiceProxy("/rviz/screenshot", Empty)
            rospy.loginfo("Connected to /rviz/screenshot")

        rospy.loginfo("RGBAndRvizScreenshot started.")
        rospy.loginfo("  topic_a: %s", self.topic_a)
        rospy.loginfo("  topic_b: %s", self.topic_b)
        rospy.loginfo("  out_dir: %s", self.out_dir)

    def _allow(self, last_t, rate_hz):
        if rate_hz <= 0.0:
            return True, last_t
        now = time.time()
        if now - last_t >= 1.0 / rate_hz:
            return True, now
        return False, last_t

    def _to_bgr_or_gray(self, msg: Image):
        """
        优先 passthrough 保持原编码；需要时再转 bgr8。[2](https://srrobot-shirui.github.io/Learning-Robotics-for-beginners/tutorial/Robot_industrial_design/06Navigation%20%26%20RViz/)
        """
        enc = (msg.encoding or "").lower()
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  # [2](https://srrobot-shirui.github.io/Learning-Robotics-for-beginners/tutorial/Robot_industrial_design/06Navigation%20%26%20RViz/)

        if enc == "bgr8":
            return cv_img
        if enc == "rgb8":
            return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        if enc in ["mono8", "mono16"]:
            return cv_img

        # 其他编码尝试强转 bgr8（可能失败则回退）
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return cv_img

    def _imwrite(self, path, img):
        if self.ext in ["jpg", "jpeg"]:
            return cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return cv2.imwrite(path, img)

    # -------- RViz window helpers (wmctrl) --------
    def _has_wmctrl(self):
        try:
            subprocess.run(["wmctrl", "-m"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def _wmctrl_restore(self):
        # 激活/恢复窗口（按标题匹配）
        subprocess.run(["wmctrl", "-R", self.rviz_window_title],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wmctrl_hide(self):
        # 通过 window id 隐藏（更稳），否则退化为最小化/隐藏
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

    def _maybe_rviz_screenshot(self):
        if not self.enable_rviz_shot or self.srv is None:
            return
        ok, newt = self._allow(self.last_rviz, self.save_rate_rviz)
        if not ok:
            return
        self.last_rviz = newt

        filename = os.path.join(self.dir_rviz, f"rviz_{ts_str()}.png")

        # 关键：最小化时截图会变桌面，所以先恢复窗口（如果用户允许且系统有wmctrl）
        if self.rviz_restore_before_shot and self._has_wmctrl():
            self._wmctrl_restore()
            time.sleep(max(self.rviz_restore_delay, 0.0))

        # 调用 service 截图。ScreenshotListenerTool 提供 /rviz/screenshot，支持传入文件名。[1](https://frankwang98.github.io/wiki/cpp_ros/cv_bridge%E5%8C%85%E4%BD%BF%E7%94%A8%E4%B8%8E%E5%9B%BE%E5%83%8F%E8%BD%AC%E6%8D%A2%E7%A4%BA%E4%BE%8B/)
        try:
            # 这个 service 的实现由插件决定；常见实现会使用你传入的 filename 保存。[1](https://frankwang98.github.io/wiki/cpp_ros/cv_bridge%E5%8C%85%E4%BD%BF%E7%94%A8%E4%B8%8E%E5%9B%BE%E5%83%8F%E8%BD%AC%E6%8D%A2%E7%A4%BA%E4%BE%8B/)
            # 在 rospy 里 std_srvs/Empty 无法传 filename，因此这里改用 rosservice 命令调用，确保带参数。
            subprocess.run(["rosservice", "call", "/rviz/screenshot", filename], check=True)
            rospy.loginfo("Saved RViz screenshot -> %s", filename)
        except Exception as e:
            rospy.logwarn("RViz screenshot failed: %s", str(e))

        if self.rviz_hide_after_shot and self._has_wmctrl():
            self._wmctrl_hide()

    # -------- callbacks --------
    def cb_a(self, msg: Image):
        ok, newt = self._allow(self.last_a, self.save_rate_img)
        if not ok:
            return
        self.last_a = newt

        try:
            img = self._to_bgr_or_gray(msg)
            path = os.path.join(self.dir_a, f"robot2_{ts_str()}.{self.ext}")
            if self._imwrite(path, img):
                rospy.loginfo("Saved robot2 -> %s", path)
            else:
                rospy.logwarn("Failed write robot2 -> %s", path)
        except Exception as e:
            rospy.logerr("cb_a error: %s", str(e))

        # 每次保存后也尝试按频率截一张 RViz
        self._maybe_rviz_screenshot()

    def cb_b(self, msg: Image):
        ok, newt = self._allow(self.last_b, self.save_rate_img)
        if not ok:
            return
        self.last_b = newt

        try:
            img = self._to_bgr_or_gray(msg)
            path = os.path.join(self.dir_b, f"robot1_{ts_str()}.{self.ext}")
            if self._imwrite(path, img):
                rospy.loginfo("Saved robot1 -> %s", path)
            else:
                rospy.logwarn("Failed write robot1 -> %s", path)
        except Exception as e:
            rospy.logerr("cb_b error: %s", str(e))

        self._maybe_rviz_screenshot()


if __name__ == "__main__":
    rospy.init_node("rgb_and_rviz_screenshot", anonymous=True)
    RGBAndRvizScreenshot()
    rospy.spin()