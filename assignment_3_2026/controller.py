# wind_flag = True

import math

# --- Outer position-loop PI gains [x, y, z, yaw] ---
# The inner TelloController already runs a velocity PID (Kp=7, Kd=0.2).
# That inner loop provides all the velocity damping needed; adding a Kd term
# here creates a large derivative spike when the error crosses zero (drone
# reaches target at speed), which fires a hard reverse command and causes
# the "shoots away from target" oscillation.  Keep this loop to P + I only.
KP = [0.40, 0.40, 0.50, 1.80]
KI_BASE = [0.03, 0.03, 0.05, 0.05]   # no-wind: small integral
KI_WIND = [0.08, 0.08, 0.10, 0.08]   # wind: larger integral for disturbance rejection

# Limits — match the simulator's check_action clip of ±1 m/s / ±1.745 rad/s
_VEL_MAX      = 1.0    # m/s
_YAW_RATE_MAX = 1.745  # rad/s  (= 100 deg/s)
_INTEG_MAX    = 1.5    # anti-windup clamp (integral units: m·s or rad·s)

# Persistent PID state
_integral    = [0.0, 0.0, 0.0, 0.0]
_last_target = None   # used to detect target changes and reset integrators


def controller(state, target_pos, dt, wind_enabled=False):
    # state:      [x(m), y(m), z(m), roll(rad), pitch(rad), yaw(rad)]  — world frame
    # target_pos: (x, y, z, yaw)
    # dt:         control timestep (s)
    # returns:    (vx, vy, vz, yaw_rate) in drone body frame (yaw-rotated, not pitch/roll)

    global _integral, _last_target

    pos_x, pos_y, pos_z, _roll, _pitch, yaw = state
    tgt_x, tgt_y, tgt_z, tgt_yaw = target_pos

    # Reset integrators whenever the target changes (e.g. arrow-key switch).
    # TelloController.reset() is called on target change but our globals persist.
    if _last_target is not None and tuple(target_pos) != _last_target:
        _integral = [0.0, 0.0, 0.0, 0.0]
    _last_target = tuple(target_pos)

    # --- Errors in world frame ---
    err_x = tgt_x - pos_x
    err_y = tgt_y - pos_y
    err_z = tgt_z - pos_z
    err_yaw = (tgt_yaw - yaw + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]

    errors = [err_x, err_y, err_z, err_yaw]

    ki = KI_WIND if wind_enabled else KI_BASE
    safe_dt = max(dt, 1e-6)
    pid_out = []

    for i in range(4):
        # Integral with anti-windup clamp
        _integral[i] = max(-_INTEG_MAX, min(_INTEG_MAX, _integral[i] + errors[i] * safe_dt))

        pid_out.append(KP[i] * errors[i] + ki[i] * _integral[i])

    # --- Saturate at simulator limits before frame rotation ---
    vx_w     = max(-_VEL_MAX,      min(_VEL_MAX,      pid_out[0]))
    vy_w     = max(-_VEL_MAX,      min(_VEL_MAX,      pid_out[1]))
    vz       = max(-_VEL_MAX,      min(_VEL_MAX,      pid_out[2]))
    yaw_rate = max(-_YAW_RATE_MAX, min(_YAW_RATE_MAX, pid_out[3]))

    # --- Rotate world-frame xy velocity into yaw-body frame ---
    # run.py passes lin_vel = rotateVector(inv_yaw_quat, lin_vel_world) to TelloController,
    # so desired_vel must be expressed in the same yaw-rotated frame.
    c, s = math.cos(yaw), math.sin(yaw)
    vx_body =  vx_w * c + vy_w * s
    vy_body = -vx_w * s + vy_w * c

    return (vx_body, vy_body, vz, yaw_rate)
