#!/usr/bin/env python3
"""
ros1_udp_bridge.py
──────────────────
A lightweight bridge to bypass Python version mismatch issues (Python 3.10 vs 3.8).
Listens for JSON corner coordinates on local UDP port 9876 and publishes them to ROS 1 Noetic.

Usage:
  1. Open a NEW terminal
  2. DO NOT ACTIVATE YOUR VIRTUAL ENVIRONMENT.
  3. Run:
     source /opt/ros/noetic/setup.bash
     python3 deployment/ros1_udp_bridge.py
"""

import rospy
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import socket
import json

def main():
    rospy.init_node('udp_to_ros1_bridge', anonymous=True)
    pub_corners = rospy.Publisher('/cloth_corners', PoseArray, queue_size=10)
    pub_mesh = rospy.Publisher('/cloth_mesh', PointCloud2, queue_size=10)
    pub_pcd = rospy.Publisher('/cloth_pcd', PointCloud2, queue_size=10)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 9876))
    sock.setblocking(False)
    
    rospy.loginfo('UDP-to-ROS1 Bridge running. Listening on port 9876...')

    rate = rospy.Rate(100) # 100Hz loop
    while not rospy.is_shutdown():
        latest_data = None
        try:
            # Drain the UDP buffer to get the latest packet
            while True:
                data, addr = sock.recvfrom(65536)
                latest_data = data
        except BlockingIOError:
            pass
            
        if latest_data is not None:
            try:
                # print(latest_data.decode("utf-8"))
                msg_dict = json.loads(latest_data.decode("utf-8"))
                
                msg = PoseArray()
                msg.header.frame_id = msg_dict.get("frame_id", "camera_depth_optical_frame")
                msg.header.stamp = rospy.Time.now()
                
                for pt in msg_dict["corners"]:
                    pose = Pose()
                    pose.position.x = float(pt[0])
                    pose.position.y = float(pt[1])
                    pose.position.z = float(pt[2])
                    pose.orientation.w = 1.0
                    msg.poses.append(pose)
                    
                pub_corners.publish(msg)
                
                if "mesh" in msg_dict:
                    header = Header()
                    header.frame_id = msg.header.frame_id
                    header.stamp = msg.header.stamp
                    pc_msg = pc2.create_cloud_xyz32(header, msg_dict["mesh"])
                    pub_mesh.publish(pc_msg)

                if "pcd" in msg_dict:
                    header = Header()
                    header.frame_id = msg.header.frame_id
                    header.stamp = msg.header.stamp
                    pc_pcd_msg = pc2.create_cloud_xyz32(header, msg_dict["pcd"])
                    pub_pcd.publish(pc_pcd_msg)
                    
            except Exception as e:
                rospy.logerr(f"Error parsing UDP data: {e}")
                
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
