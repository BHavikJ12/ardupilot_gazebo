from pymavlink import mavutil
import math
from scipy.spatial.transform import Rotation as R
import numpy as np

# Connect to MAVLink (you may already have this in your code)
master = mavutil.mavlink_connection('udp:127.0.0.1:14550')  # or your telemetry port
master.wait_heartbeat()
print("Heartbeat received!")

# Define Euler angles
pitch = math.radians(-45.0)  # degrees to radians
yaw = math.radians(0.0)
roll = math.radians(0.0)

# Convert to quaternion (w, x, y, z)
rotation = R.from_euler('xyz', [roll, pitch, yaw])
q = rotation.as_quat()  # [x, y, z, w]
q_mav = [q[3], q[0], q[1], q[2]]  # reorder to [w, x, y, z]

# Send GIMBAL_MANAGER_SET_ATTITUDE
master.mav.gimbal_manager_set_attitude_send(
    target_system=1,
    target_component=1,              # Usually autopilot component is 1
    flags=0b00000000,                # GIMBAL_MANAGER_FLAGS: 0 = no lock, full attitude control
    gimbal_device_id=0,              # 0 for all gimbals
    q=q_mav,                         # Quaternion [w, x, y, z]
    angular_velocity_x=float('nan'),  # NaN to ignore rate
    angular_velocity_y=float('nan'),
    angular_velocity_z=float('nan')
)
