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
        threading.Timer(5.0, self.run_startup_thread).start()
       
        # self.trk = cv2.TrackerKCF_create()    # only kcf
        self.kcf = cv2.TrackerKCF_create()      # fused kcf & csrt
        self.csrt = cv2.TrackerCSRT_create()    # fused kcf & csrt
        self.frame_count = 0                    # fused kcf & csrt
        self.roi_selected = False
        self.prev_time = time.time()
        self.fps = 0.0
        self.awaiting_roi = False
        self.prev_centroid = None

        self.centering_err = (0,0)
        self.motion_err = (0,0)     
        self.kp_x, self.ki_x, self.kd_x = 0.1, 0.000, 0.03
        self.kp_y, self.ki_y, self.kd_y = 0.1, 0.000, 0.03
        self._int_x = self._int_y = 0.0
        self._prev_err_x = self._prev_err_y = 0.0
        self.pid_dt = 1.0 / 30.0  # 30 Hz
        self.create_timer(self.pid_dt, self.pid_loop)

        self.target_acquired = False
        self.mode_switched = False


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
    
    def set_param(self,param_id, param_value, param_type):
        print(f"Setting {param_id} to {param_value}")
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            param_id.encode('utf-8'),
            float(param_value),
            param_type
        )
        while True:
            msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
            if msg and msg.param_id.strip('\x00') == param_id:
                print(f"Confirmed: {msg.param_id} = {msg.param_value}")
                break

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

        # if not self.set_flight_mode("GUIDED"):
        #     self.get_logger().error("Failed to set GUIDED mode.")
        #     return

        # if not self.arm():
        #     self.get_logger().error("Failed to arm.")
        #     return

        # if not self.takeoff(10.0):
        #     self.get_logger().error("Failed to initiate take‑off.")
            return

        self.set_param("ANGLE_MAX", 8000, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
        self.set_param("ATC_ACCEL_P_MAX", 110000.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

        self.set_flight_mode("GUIDED")
        self.arm()
        self.takeoff(10.0)

        self.set_gimbal_angles(pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0)
 
        self.get_logger().info("Startup sequence complete ✔️")
        self._startup_done = True

        # time.sleep(20)
        # print("stabilize")
        # self.set_flight_mode("STABILIZE")
        # if not self.set_flight_mode("STABILIZE"):
        #     self.get_logger().error("GUIDED→STABILIZE failed.")


    def run_startup_thread(self):
        if self._startup_done:
            return
        threading.Thread(target=self.execute, daemon=True).start()

    def trackerkcf(self, frame: np.ndarray):
        """Run / update KCF tracker on the incoming frame."""
        # Wait until SPACE bar is pressed to start ROI selection
        if not self.roi_selected and self.awaiting_roi:
            bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            self.awaiting_roi = False  # Reset flag
            if bbox[2] > 0 and bbox[3] > 0:
                self.trk.init(frame, bbox)
                self.roi_selected = True
            return  # Don't proceed with tracking until next frame

        if not self.roi_selected:
            return  # ROI not selected and space not pressed → skip frame

        # Update tracker
        success, bbox = self.trk.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        # FPS counter
        # now = time.time()
        # self.fps = 1.0 / (now - self.prev_time)
        # self.prev_time = now
        # cv2.putText(frame, f"FPS: {self.fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (255, 255, 0), 2)

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)


    def tracker_fused(self, frame: np.ndarray):
        if not self.roi_selected and self.awaiting_roi:
            bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            self.awaiting_roi = False
            if bbox[2] > 0 and bbox[3] > 0:
                self.kcf = cv2.TrackerKCF_create()
                self.csrt = cv2.TrackerCSRT_create()
                self.kcf.init(frame, bbox)
                self.csrt.init(frame, bbox)
                self.roi_selected = True
            return

        if not self.roi_selected:
            return

        self.frame_count += 1

        ok_kcf, bbox_kcf = self.kcf.update(frame)

        if ok_kcf:

            if not self.target_acquired:
                self.target_acquired = True     # latch it
                self.get_logger().info("🎯 Target locked – requesting STABILIZE mode")
                self.set_flight_mode("STABILIZE")
                # if self.set_flight_mode("STABILIZE"):
                #     self.mode_switched = True
                # else:
                #     self.get_logger().error("Could not switch to STABILIZE")


            x, y, w, h = map(int, bbox_kcf)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking (KCF)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---------- NEW: centroid + errors ----------
            cx, cy = x + w / 2.0, y + h / 2.0          # centroid
            img_h, img_w = frame.shape[:2]

            self.centering_err  = (cx - img_w / 2.0, cy - img_h / 2.0)
            self.motion_err     = (0.0, 0.0)
            if self.prev_centroid is not None:
                self.motion_err = (cx - self.prev_centroid[0],
                            cy - self.prev_centroid[1])
            self.prev_centroid = (cx, cy)

            # Optional visual overlay  (green = centering, yellow = motion)
            cv2.drawMarker(frame, (int(cx), int(cy)), (0, 255, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
            cv2.putText(frame,
                        f"ctr_err: ({self.centering_err[0]:+.1f}, {self.centering_err[1]:+.1f})",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"mot_err: ({self.motion_err[0]:+.1f}, {self.motion_err[1]:+.1f})",
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # FPS Counter
        # now = time.time()
        # self.fps = 1.0 / (now - self.prev_time)
        # self.prev_time = now
        # cv2.putText(frame, f"FPS: {self.fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (255, 255, 0), 2)

    def pid(self, err, axis):
        if axis == 'x':                               # roll
            kp, ki, kd = self.kp_x, self.ki_x, self.kd_x
            self._int_x += err * self.pid_dt
            deriv = (err - self._prev_err_x) / self.pid_dt
            self._prev_err_x = err
            #  ➜ pid_out is in “pixel error × gain” units
            pid_out = kp * err + ki * self._int_x + kd * deriv
            #  ➜ map to RC 1000‑2000 around centre 1500
            return int(np.clip(1500 + pid_out * 100, 1000, 2000))

        else:                                         # pitch (axis 'y')
            kp, ki, kd = self.kp_y, self.ki_y, self.kd_y
            self._int_y += err * self.pid_dt
            deriv = (err - self._prev_err_y) / self.pid_dt
            self._prev_err_y = err
            pid_out = kp * err + ki * self._int_y + kd * deriv
            # invert sign because image Y grows downward
            return int(np.clip(1500 - pid_out * 100, 1000, 2000))


    def pid_loop(self):
        if not (self._startup_done and self.roi_selected):
            return

        roll_cmd  = self.pid(self.centering_err[0], 'x')   # CH1
        pitch_cmd = self.pid(self.centering_err[1], 'y')   # CH2
        print(pitch_cmd)
        print(roll_cmd)
        throttle_cmd = 1200                                # CH3 (hold)
        yaw_cmd      = 1500                                # CH4 (hold)

        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            roll_cmd, pitch_cmd, throttle_cmd, yaw_cmd,
            0, 0, 0, 0
        )


    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Run tracker & draw overlays
            self.tracker_fused(frame)

            # Display
            cv2.imshow("KCF Visual Guidance", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # SPACE bar
                self.awaiting_roi = True

            if key == ord("q"):
                rclpy.shutdown()


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
