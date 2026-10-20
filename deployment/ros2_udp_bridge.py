#!/usr/bin/env python3
"""
ros2_udp_bridge.py
──────────────────
A lightweight bridge to bypass Python version mismatch issues (Python 3.10 vs 3.8).
Listens for JSON corner coordinates on local UDP port 9876 and publishes them to ROS 2.

Usage:
  1. Open a NEW terminal
  2. DO NOT ACTIVATE YOUR VIRTUAL ENVIRONMENT.
  3. Run:
     source /opt/ros/foxy/setup.bash
     python3 deployment/ros2_udp_bridge.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import socket
import json

class UDPtoROSBridge(Node):
    def __init__(self):
        super().__init__('udp_to_ros_bridge')
        self.publisher_ = self.create_publisher(PoseArray, '/cloth_corners', 10)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 9876))
        self.sock.setblocking(False)
        
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz check
        self.get_logger().info('UDP-to-ROS Bridge running. Listening on port 9876...')

    def timer_callback(self):
        try:
            # Drain the UDP buffer to get the latest packet
            while True:
                data, addr = self.sock.recvfrom(4096)
                
        except BlockingIOError:
            # When BlockingIOError happens, `data` contains the last read packet
            try:
                msg_dict = json.loads(data.decode("utf-8"))
                
                msg = PoseArray()
                msg.header.frame_id = msg_dict.get("frame_id", "camera_depth_optical_frame")
                msg.header.stamp = self.get_clock().now().to_msg()
                
                for pt in msg_dict["corners"]:
                    pose = Pose()
                    pose.position.x = float(pt[0])
                    pose.position.y = float(pt[1])
                    pose.position.z = float(pt[2])
                    pose.orientation.w = 1.0
                    msg.poses.append(pose)
                    
                self.publisher_.publish(msg)
            except NameError:
                pass # data is not defined yet (buffer is completely empty)
            except Exception as e:
                self.get_logger().error(f"Error parsing UDP data: {e}")

def main(args=None):
    rclpy.init(args=args)
    bridge = UDPtoROSBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
