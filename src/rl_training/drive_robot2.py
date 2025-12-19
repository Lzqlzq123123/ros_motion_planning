import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import random

def move_robot2():
    """
    Move robot2 to random locations in the warehouse.
    用于自动发送robot2的随机目标点
    """
    rospy.init_node('robot2_random_patrol')
    
    client = actionlib.SimpleActionClient('/robot2/move_base', MoveBaseAction)
    rospy.loginfo("Waiting for robot2 move_base...")
    client.wait_for_server()
    
    while not rospy.is_shutdown():
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Random goal in warehouse
        goal.target_pose.pose.position.x = random.uniform(-5, 5)
        goal.target_pose.pose.position.y = random.uniform(-5, 5)
        goal.target_pose.pose.orientation.w = 1.0
        
        rospy.loginfo(f"Sending goal to robot2: {goal.target_pose.pose.position.x}, {goal.target_pose.pose.position.y}")
        client.send_goal(goal)
        client.wait_for_result()
        
if __name__ == '__main__':
    try:
        move_robot2()
    except rospy.ROSInterruptException:
        pass
