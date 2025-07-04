#!/usr/bin/env python3
from pymavlink import mavutil
import time

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

    # Wait for ACK
    while True:
        msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
        if msg and msg.param_id.strip('\x00') == param_id:
            print(f"Confirmed: {msg.param_id} = {msg.param_value}")
            break

# Set ANGLE_MAX (e.g., 3000 = 30.00 degrees in centidegrees)
set_param("ANGLE_MAX", 8000, mavutil.mavlink.MAV_PARAM_TYPE_INT32)

# Set ATC_ACCEL_P_MAX (e.g., 110000 = 110000 deg/s^2)
set_param("ATC_ACCEL_P_MAX", 110000.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

# Set mode to STABILIZE
def set_mode_stabilize():
    mode_id = master.mode_mapping()['STABILIZE']
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
        0, 0, 0, 0, 0
    )
    print("Setting mode to STABILIZE...")
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
    print(f"ACK: {ack}")

# Arm the drone
def arm():
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,  # Arm
        0, 0, 0, 0, 0, 0
    )
    print("Arming drone...")
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
    print(f"ACK: {ack}")

set_mode_stabilize()
arm()

# RC channels: 1=roll, 2=pitch, 3=throttle, 4=yaw
# Override values: 1000-2000, 1500 = neutral
# Example: forward pitch and medium throttle
RC_ROLL     = 1500  # no roll
RC_PITCH    = 1150  # forward pitch
RC_THROTTLE = 1600  # medium throttle
RC_YAW      = 1500  # no yaw

print("Sending continuous RC overrides... Press Ctrl+C to stop.")

try:
    while True:
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            RC_ROLL,     # CH1: Roll
            RC_PITCH,    # CH2: Pitch
            RC_THROTTLE, # CH3: Throttle
            RC_YAW,      # CH4: Yaw
            0, 0, 0, 0   # CH5–CH8: 0 disables override
        )
        time.sleep(0.1)  # Send at 10Hz (recommended)
except KeyboardInterrupt:
    print("\nStopping RC override and exiting...")
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )
