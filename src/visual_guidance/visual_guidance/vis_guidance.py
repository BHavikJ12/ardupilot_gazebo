#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
from pymavlink import mavutil
import numpy as np

class VisualGuidanceNode(Node):
    def __init__(self):
        super().__init__('visual_guidance_node')
        self.bridge = CvBridge()

        self.master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
        self.master.wait_heartbeat()
        self.get_logger().info("Connected to MAVLink")

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw', 
            self.image_callback,
            10
        )

        self.timer = self.create_timer(1.0, self.check_gimbal_status)

        self.use_gimbal_manager = True

        self.send_gimbal_attitude(pitch_deg=-45.0, roll_deg=0.0, yaw_deg=0.0)

    def send_gimbal_attitude(self, pitch_deg, roll_deg, yaw_deg):
        """Send gimbal attitude command, trying GIMBAL_MANAGER_SET_ATTITUDE first, then falling back to MAV_CMD_DO_MOUNT_CONTROL."""
        if self.use_gimbal_manager:
            try:
                pitch = math.radians(pitch_deg)
                roll = math.radians(roll_deg)
                yaw = math.radians(yaw_deg)

                from scipy.spatial.transform import Rotation as R
                quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
                q_mav = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])] 

                self.master.mav.gimbal_manager_set_attitude_send(
                    target_system=1,  
                    target_component=154,  
                    flags=0, 
                    gimbal_device_id=0, 
                    q=q_mav, 
                    angular_velocity_x=float('nan'),
                    angular_velocity_y=float('nan'),
                    angular_velocity_z=float('nan')
                )
                self.get_logger().info(f"Sent gimbal attitude (GIMBAL_MANAGER): pitch={pitch_deg}, roll={roll_deg}, yaw={yaw_deg}")
            except Exception as e:
                self.get_logger().error(f"GIMBAL_MANAGER_SET_ATTITUDE failed: {e}. Switching to MAV_CMD_DO_MOUNT_CONTROL.")
                self.use_gimbal_manager = False

        # if not self.use_gimbal_manager:
        #     try:
        #         self.master.mav.command_long_send(
        #             target_system=1,
        #             target_component=0,  
        #             command=mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
        #             confirmation=0,
        #             param1=pitch_deg,
        #             param2=roll_deg,
        #             param3=yaw_deg,
        #             param4=0,
        #             param5=0,
        #             param6=0,
        #             param7=mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
        #         )
        #         self.get_logger().info(f"Sent gimbal attitude (MOUNT_CONTROL): pitch={pitch_deg}, roll={roll_deg}, yaw={yaw_deg}")
        #     except Exception as e:
        #         self.get_logger().error(f"Failed to send gimbal command (MOUNT_CONTROL): {e}")

    def check_gimbal_status(self):
        """Check gimbal status by listening to GIMBAL_DEVICE_ATTITUDE_STATUS."""
        try:
            msg = self.master.recv_match(type='GIMBAL_DEVICE_ATTITUDE_STATUS', blocking=False)
            if msg:
                from scipy.spatial.transform import Rotation as R
                q = [msg.q[1], msg.q[2], msg.q[3], msg.q[0]]  # Reorder [w, x, y, z] to [x, y, z, w]
                euler = R.from_quat(q).as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f"Gimbal status: pitch={euler[1]:.2f}, roll={euler[0]:.2f}, yaw={euler[2]:.2f}"
                )
                if abs(euler[1] - (-45.0)) > 1.0 or abs(euler[0] - 0.0) > 1.0 or abs(euler[2] - 0.0) > 1.0:
                    self.get_logger().warn("Gimbal status does not match commanded values")
            else:
                self.get_logger().info("No gimbal status received")
        except Exception as e:
            self.get_logger().error(f"Error checking gimbal status: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Camera Feed", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

    def destroy_node(self):
        """Cleanup on node shutdown."""
        super().destroy_node()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = VisualGuidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()