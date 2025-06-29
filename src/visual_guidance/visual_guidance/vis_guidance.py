#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        super().__init__("visual_guidance_node")
        self.bridge = CvBridge()

        self.master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
        self.master.wait_heartbeat()
        self.get_logger().info(
            f"Connected to MAVLink (sys:{self.master.target_system}, "
            f"comp:{self.master.target_component})"
        )

        self.subscription = self.create_subscription(
            Image, "/camera/image", self.image_callback, 10
        )

        self._initial_cmd_sent = False

    def set_gimbal_angles(
        self,
        pitch_deg: float,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
        mount_mode: int = mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING,
    ):
        """

        Args
        ----
        pitch_deg : float – positive looks **up**, negative looks **down**
        roll_deg  : float – roll in degrees (normally 0)
        yaw_deg   : float – 0 = vehicle heading; +right / ‑left
        mount_mode: int   – one of MAV_MOUNT_MODE_*
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
            0,               
            pitch_deg,        
            roll_deg,         
            yaw_deg,          
            0, 0, 0,          
            mount_mode,       
        )
        self.get_logger().info(
            f"[GIMBAL] pitch={pitch_deg:.1f}°, roll={roll_deg:.1f}°, yaw={yaw_deg:.1f}°"
        )
    def image_callback(self, msg: Image):
        """
        Demo behaviour:
        • Shows the live camera feed.
        • Sends one gimbal command at startup (‑50° pitch, 0° yaw).
        Replace this with your own vision/target‑tracking logic:
        call `set_gimbal_angles()` whenever you need to steer the gimbal.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)

            if not self._initial_cmd_sent:
                self.set_gimbal_angles(pitch_deg=-45.0, roll_deg=0.0, yaw_deg=0.0)
                self._initial_cmd_sent = True

        except Exception as err:
            self.get_logger().error(f"image_callback error: {err}")


    def destroy_node(self):
        super().destroy_node()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = VisualGuidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
