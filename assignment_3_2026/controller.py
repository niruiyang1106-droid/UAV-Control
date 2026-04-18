# wind_flag = True

import math

# --- PID gains [x, y, z, yaw] ---
KP = [1.0, 1.0, 1.2, 2.0]
KI = [0.05, 0.05, 0.10, 0.05]
KD = [0.50, 0.50, 0.60, 0.30]

# Output saturation limits
_VEL_MAX     = 2.0   # m/s
_YAW_RATE_MAX = 1.5  # rad/s
_INTEG_MAX   = 2.0   # anti-windup clamp

# Persistent PID state (reset externally by re-importing or via simulator 'r' key)
_integral  = [0.0, 0.0, 0.0, 0.0]  # accumulated integral terms [x, y, z, yaw]
_prev_err  = [0.0, 0.0, 0.0, 0.0]  # previous errors for derivative [x, y, z, yaw]
_first_call = True


def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (rad), pitch (rad), yaw (rad)]
    # target_pos format: (x (m), y (m), z (m), yaw (rad))
    # dt: time step (s)
    # wind_enabled: boolean flag for wind disturbance consideration
    # return: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (rad/s))

    global _integral, _prev_err, _first_call

    pos_x, pos_y, pos_z, _roll, _pitch, yaw = state
    tgt_x, tgt_y, tgt_z, tgt_yaw = target_pos

    # --- Compute world-frame position errors ---
    err_x = tgt_x - pos_x
    err_y = tgt_y - pos_y
    err_z = tgt_z - pos_z

    # Yaw error wrapped to [-pi, pi]
    err_yaw = tgt_yaw - yaw
    err_yaw = (err_yaw + math.pi) % (2 * math.pi) - math.pi

    errors = [err_x, err_y, err_z, err_yaw]

    if _first_call:
        # Initialise derivative state to current errors to avoid a spike on first call
        _prev_err = errors[:]
        _first_call = False

    # --- PID computation for each axis ---
    safe_dt = max(dt, 1e-6)
    pid_out = []
    for i in range(4):
        # Integral with anti-windup clamp
        _integral[i] += errors[i] * safe_dt
        _integral[i] = max(-_INTEG_MAX, min(_INTEG_MAX, _integral[i]))

        # Derivative (backward difference)
        deriv = (errors[i] - _prev_err[i]) / safe_dt

        pid_out.append(KP[i] * errors[i] + KI[i] * _integral[i] + KD[i] * deriv)

    _prev_err = errors[:]

    # --- Saturate outputs ---
    vx_w = max(-_VEL_MAX,      min(_VEL_MAX,      pid_out[0]))
    vy_w = max(-_VEL_MAX,      min(_VEL_MAX,      pid_out[1]))
    vz   = max(-_VEL_MAX,      min(_VEL_MAX,      pid_out[2]))
    yaw_rate = max(-_YAW_RATE_MAX, min(_YAW_RATE_MAX, pid_out[3]))

    # --- Transform world-frame xy velocity to drone body frame (yaw rotation only) ---
    # The simulator control frame is not rotated in pitch/roll, only in yaw.
    # R_z(-yaw) maps world velocities to body velocities.
    c, s = math.cos(yaw), math.sin(yaw)
    vx_body =  vx_w * c + vy_w * s
    vy_body = -vx_w * s + vy_w * c

    return (vx_body, vy_body, vz, yaw_rate)
