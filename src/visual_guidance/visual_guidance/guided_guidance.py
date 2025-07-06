#!/usr/bin/env python3
from pymavlink import mavutil
import time
import math

# Connect to the vehicle
master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
master.wait_heartbeat()
print(f"Connected to system {master.target_system}, component {master.target_component}")

def set_param(param_id, param_value, param_type):
    print(f"Setting {param_id} to {param_value}")
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        param_id.encode('utf-8'),
        float(param_value),
        param_type
    )

    while True:
        msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
        if msg and msg.param_id.strip('\x00') == param_id:
            print(f"Confirmed: {msg.param_id} = {msg.param_value}")
            break

def set_mode_stabilize():
    mode_id = master.mode_mapping()['GUIDED']
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
        0, 0, 0, 0, 0
    )
    print("Setting mode to GUIDED...")
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
    print(f"ACK: {ack}")

def arm():
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 
        0, 0, 0, 0, 0, 0
    )
    print("Arming drone...")
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
    print(f"ACK: {ack}")

def receive_messages(msg_types=None, timeout=10):
    # start_time = time.time()
    print(f"Listening for messages... (timeout: {timeout}s)")
    
    # while time.time() - start_time < timeout:
    msg = master.recv_match(blocking=False, timeout=1)
    print(msg)
        # if msg:
        #     if not msg_types or msg.get_type() in msg_types:
        #         print(f"[{msg.get_type()}] {msg.to_dict()}")
        # else:
        #     print("No message received within timeout window.")

    # if msg:
    #     roll = math.degrees(msg.roll)
    #     pitch = math.degrees(msg.pitch)
    #     yaw = math.degrees(msg.yaw)
    #     print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
    # else:
    #     print("No ATTITUDE message received.")
    

def set_gimbal_angles(pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0,
                      mount_mode=mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING):
    print("Sending gimbal control command...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
        0,                # Confirmation
        pitch_deg,        # Param 1: Pitch (deg)
        roll_deg,         # Param 2: Roll (deg)
        yaw_deg,          # Param 3: Yaw (deg)
        0, 0, 0,          # Params 4–6: Unused
        mount_mode        # Param 7: MAV_MOUNT_MODE
    )

    print(f"[GIMBAL] Pitch={pitch_deg:.1f}°, Roll={roll_deg:.1f}°, Yaw={yaw_deg:.1f}°")

def takeoff(altitude=10.0, max_retries=1, ack_timeout=8.0):
    for attempt in range(1, max_retries + 1):
        print(f"Takeoff attempt {attempt} to {altitude:.1f} meters...")

        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,                 # Confirmation
            0, 0, 0,           # Params 1–3: pitch, yaw, etc. (ignored)
            float('nan'),      # Yaw angle: NaN = unchanged
            0, 0,              # Latitude, Longitude: NaN = current pos
            altitude           # Altitude (relative to home)
        )

        ack = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=ack_timeout)

        if not ack or ack.command != mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            print("[WARN] No ACK for takeoff.")
        elif ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print(f"Takeoff command accepted to {altitude:.1f} m.")
            return True
        else:
            print(f"[WARN] Takeoff rejected (result={ack.result})")

def send_position_velocity_target_global_int(lat_deg, lon_deg, alt_m,
                                             vx=0.0, vy=0.0, vz=0.0, ax=0.0, ay=0.0, az=0.0,
                                             yaw_deg=None, yaw_rate=None):
    # Convert lat/lon to degrees * 1e7
    lat_int = int(lat_deg * 1e7)
    lon_int = int(lon_deg * 1e7)

    # Use GLOBAL_RELATIVE_ALT so altitude is relative to home
    coordinate_frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT

    # Type mask: ignore acceleration and force fields
    # 0b0000111111000111 = 0x0FFF7 = 3575
 
    type_mask = 0

    if yaw_deg is not None:
        type_mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE

    if yaw_rate is not None:
        type_mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

    # Convert yaw to radians
    yaw_rad = math.radians(yaw_deg) if yaw_deg is not None else 0
    yaw_rate_rad = yaw_rate if yaw_rate is not None else 0

    master.mav.set_position_target_global_int_send(
        int(time.time()*1000) & 0xFFFFFFFF,     # time_boot_ms
        master.target_system,
        master.target_component,
        coordinate_frame,
        type_mask,
        lat_int,
        lon_int,
        alt_m,
        vx,vy,vz,
        ax,ay,az,                                                                                  # acceleration (ignored)
        yaw_rad,
        yaw_rate_rad
    )

    # print(f"[POS‑VEL] lat={lat_deg:.7f}, lon={lon_deg:.7f}, alt={alt_m:.1f} m | "
    #       f"v(N,E,D)=({vx},{vy},{vz}) m/s | a(N,E,D)=({ax},{ay},{az}) m/s²")


# receive_messages(msg_types=["GLOBAL_POSITION_INT"], timeout=5)
# Set aggressive tuning parameters
set_param("PSC_ANGLE_MAX", 0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)         # Use ANGLE_MAX instead
set_param("PSC_POSZ_P", 2.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_VELZ_P", 5.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_VELZ_FF", 0.8, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_ACCZ_P", 0.5, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_ACCZ_FF", 0.5, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

set_param("PSC_POSXY_P", 1.5, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_VELXY_P", 5.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_VELXY_FF", 5.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

set_param("PSC_JERK_XY", 15.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
set_param("PSC_JERK_Z", 45.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

set_param("ANGLE_MAX", 8000, mavutil.mavlink.MAV_PARAM_TYPE_INT32)           # 80° in centidegrees

set_param("ATC_ACCEL_P_MAX", 162000, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)  # High pitch acceleration
set_param("ATC_RATE_P_MAX", 360, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)    # Max pitch rate
set_param("ATC_ANG_PIT_P", 10.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
# set_param("ATC_RATE_P_FF", 0.8, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)       # Pitch rate feedforward

set_param("WPNAV_SPEED", 2000, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)        # Horizontal speed (cm/s)
set_param("WPNAV_ACCEL", 500, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)        # XY acceleration (cm/s²)
set_param("WPNAV_SPEED_DN", 500, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)      # Descent speed (cm/s)
set_param("WPNAV_ACCEL_Z", 500, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)       # Z-axis acceleration (cm/s²)
set_param("WPNAV_JERK", 15, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

set_mode_stabilize()
arm()
takeoff()
set_gimbal_angles()
time.sleep(10)


while True:
    send_position_velocity_target_global_int(
        lat_deg=-35.3631723,
        lon_deg=149.1652375,
        alt_m=1.0,
        vx=0.0, vy=0.0, vz=10.0,
        ax=0.0 ,ay=0.0, az=5.0,
        yaw_deg=0.0  # Optional
    )
    receive_messages(msg_types=["ATTITUDE"])
    time.sleep(0.03)


# 'lat': -353631723, 'lon': 1491652375