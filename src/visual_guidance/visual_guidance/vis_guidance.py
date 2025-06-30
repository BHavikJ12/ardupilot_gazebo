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
import time
import threading

class VisualGuidanceNode(Node):
    def __init__(self):
        super().__init__("visual_guidance_node")
        self.bridge = CvBridge()

        self.master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
        self.master.wait_heartbeat()
        self.get_logger().info(
            f"Connected to MAVLink (sys:{self.master.target_system}, "
            f"comp:{self.master.target_component})")

        self.subscription = self.create_subscription(
            Image, "/camera/image", self.image_callback, 10
        )

        self._startup_done = False
        self._startup_timer = self.create_timer(1.0, self.run_startup_thread)



    def set_flight_mode(self, mode: str = "STABILIZE",
                        max_retries: int = 1,
                        ack_timeout: float = 1.0) -> bool:
        
        mode_map = self.master.mode_mapping()
        if mode not in mode_map:
            self.get_logger().error(f"Unknown mode: {mode}")
            return False
        mode_id = mode_map[mode]

        for attempt in range(1, max_retries + 1):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,   # confirmation
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
                0, 0, 0, 0, 0)

            ack = self.master.recv_match(
                type="COMMAND_ACK",
                blocking=True,
                timeout=ack_timeout)

            if not ack or ack.command != mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                self.get_logger().warning("[WARN] No COMMAND_ACK, retrying…")
            elif ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.get_logger().info(f"Flight mode set to {mode}")
                return True
            else:
                self.get_logger().warning(
                    f"[WARN] Mode change rejected (result={ack.result}), retrying…")

            time.sleep(0.5)

        self.get_logger().error("[ERROR] Failed to change mode after retries.")
        return False
    
    def arm(self, max_retries: int = 1, ack_timeout: float = 1.0) -> bool:
        for attempt in range(1, max_retries + 1):
            self.get_logger().info(f"Arming attempt {attempt}...")

            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # Confirmation
                1,  # 1 to arm
                0, 0, 0, 0, 0, 0)

            ack = self.master.recv_match(
                type="COMMAND_ACK",
                blocking=True,
                timeout=ack_timeout)

            if not ack or ack.command != mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                self.get_logger().warning("[WARN] No ACK for arming, retrying...")
            elif ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.get_logger().info("Drone armed successfully.")
                return True
            else:
                self.get_logger().warning(f"[WARN] Arming rejected (result={ack.result}), retrying...")

            time.sleep(0.5)

        self.get_logger().error("Failed to arm drone after retries.")
        return False
    
    def disarm(self, max_retries: int = 1, ack_timeout: float = 1.0) -> bool:

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(f"Disarming attempt {attempt}...")

            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # Confirmation
                0,  # 0 to disarm
                0, 0, 0, 0, 0, 0
            )

            ack = self.master.recv_match(
                type="COMMAND_ACK",
                blocking=True,
                timeout=ack_timeout)

            if not ack or ack.command != mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                self.get_logger().warning("[WARN] No ACK for disarming, retrying...")
            elif ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.get_logger().info("Drone disarmed successfully.")
                return True
            else:
                self.get_logger().warning(f"[WARN] Disarm rejected (result={ack.result}), retrying...")

            time.sleep(0.5)

        self.get_logger().error("Failed to disarm drone after retries.")
        return False
    
    def takeoff(self, altitude: float = 10.0, max_retries: int = 1, ack_timeout: float = 3.0) -> bool:

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(f"Takeoff attempt {attempt} to {altitude:.1f} m")

            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,         # Confirmation
                0, 0, 0,   # pitch, yaw, empty params
                float('nan'),  # yaw angle (NaN = unchanged)
                0, 0,      # latitude, longitude (NaN = current position)
                altitude   # target altitude (relative to home)
            )

            ack = self.master.recv_match(
                type="COMMAND_ACK",
                blocking=True,
                timeout=ack_timeout)

            if not ack or ack.command != mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                self.get_logger().warning("[WARN] No ACK for takeoff, retrying...")
            elif ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.get_logger().info(f"Takeoff initiated to {altitude:.1f} m.")
                return True
            else:
                self.get_logger().warning(f"[WARN] Takeoff rejected (result={ack.result}), retrying...")

            time.sleep(0.5)

        self.get_logger().error("Failed to initiate takeoff after retries.")
        return False

    def set_gimbal_angles(
        self,
        pitch_deg: float,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
        mount_mode: int = mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING):
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
            mount_mode)
        self.get_logger().info(f"[GIMBAL] pitch={pitch_deg:.1f}°, roll={roll_deg:.1f}°, yaw={yaw_deg:.1f}°")

    def execute(self):
        if self._startup_done:
            return 

        if not self.set_flight_mode("GUIDED"):
            self.get_logger().error("Failed to set GUIDED mode.")
            return

        if not self.arm():
            self.get_logger().error("Failed to arm.")
            return

        if not self.takeoff(5.0):
            self.get_logger().error("Failed to initiate take‑off.")
            return

        self.set_gimbal_angles(pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0)

        self.get_logger().info("Startup sequence complete ✔️")
        self._startup_done = True
        self._startup_timer.cancel() 

    def run_startup_thread(self):
        if self._startup_done:
            return
        threading.Thread(target=self.execute, daemon=True).start()


    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)

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
