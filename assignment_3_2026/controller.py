import math

# =====================================================================
# OUTER POSITION CONTROLLER  (runs at 20 Hz via run.py)
#
# This is the "outer loop" of a cascaded control system:
#   Outer loop (HERE):  position error  →  velocity setpoint
#   Inner loop (tello_controller.py):  velocity setpoint  →  motor RPMs
#
# For each axis we use a PI controller:
#   P (Proportional): react immediately to the current position error
#   I (Integral):     slowly correct any small residual offset
#
# The output is a 4-element tuple: (vx, vy, vz, yaw_rate)
# expressed in the drone's yaw-body frame (not world frame).
# =====================================================================

# --- Proportional gains [x, y, z, yaw] ---
# Increase to fly toward the target faster and hold position more tightly.
# Too large → oscillation.
KP = [0.7, 0.7, 0.9, 2.0]

# --- Integral gains [x, y, z, yaw] ---
# Corrects small steady-state offsets (e.g. from imperfect inner loop or wind).
KI = [0.05, 0.05, 0.08, 0.05]

# --- Speed limits (must match simulator's ±1 m/s / ±1.745 rad/s clip) ---
VEL_MAX = 1.0    # m/s
YAW_MAX = 1.745  # rad/s

# --- Anti-windup clamp ---
# Prevents the integral from growing unboundedly if the drone is stuck.
INTEG_MAX = 1.5  # m·s (or rad·s for yaw)

# --- Conditional integration distance ---
# Only accumulate the x/y/z integral when within this distance of the target.
# This stops "integral wind-up" during the long approach flight, which would
# otherwise push the drone past the target and cause landing offset.
CLOSE_DIST = 0.5  # metres

# --- Persistent state (survives between controller calls) ---
_integral    = [0.0, 0.0, 0.0, 0.0]
_last_target = None


def controller(state, target_pos, dt, wind_enabled=False):
    # state:      [x, y, z, roll, pitch, yaw]  world frame (m / rad)
    # target_pos: (x, y, z, yaw)
    # dt:         control timestep (s)
    # returns:    (vx, vy, vz, yaw_rate) in drone yaw-body frame

    global _integral, _last_target

    # Unpack state and target
    x, y, z, _roll, _pitch, yaw = state
    tx, ty, tz, t_yaw = target_pos

    # Reset integral whenever the target changes
    if _last_target is not None and tuple(target_pos) != _last_target:
        _integral = [0.0, 0.0, 0.0, 0.0]
    _last_target = tuple(target_pos)

    # ------------------------------------------------------------------
    # Step 1 — Position errors in the world frame
    # ------------------------------------------------------------------
    ex    = tx - x
    ey    = ty - y
    ez    = tz - z
    # Wrap yaw error to [-π, π] so the drone always turns the short way
    e_yaw = (t_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

    errors = [ex, ey, ez, e_yaw]

    # ------------------------------------------------------------------
    # Step 2 — Update the integrals (with anti-windup clamp)
    #
    # For x/y/z: only integrate when close to the target.
    #   Reason: if we integrate during the full 3 m approach, the integral
    #   winds up a large value that overshoots the landing point.
    # For yaw: always integrate (yaw has no "distance" to worry about).
    # ------------------------------------------------------------------
    safe_dt     = max(dt, 1e-6)
    dist_to_tgt = math.sqrt(ex**2 + ey**2 + ez**2)

    for i in range(4):
        if i == 3 or dist_to_tgt < CLOSE_DIST:          # yaw, or close to target
            _integral[i] += errors[i] * safe_dt
            _integral[i]  = max(-INTEG_MAX, min(INTEG_MAX, _integral[i]))

    # ------------------------------------------------------------------
    # Step 3 — PI control → world-frame velocity commands
    # ------------------------------------------------------------------
    vx       = KP[0] * ex    + KI[0] * _integral[0]
    vy       = KP[1] * ey    + KI[1] * _integral[1]
    vz       = KP[2] * ez    + KI[2] * _integral[2]
    yaw_rate = KP[3] * e_yaw + KI[3] * _integral[3]

    # ------------------------------------------------------------------
    # Step 4 — Clamp to simulator speed limits
    # ------------------------------------------------------------------
    vx       = max(-VEL_MAX, min(VEL_MAX, vx))
    vy       = max(-VEL_MAX, min(VEL_MAX, vy))
    vz       = max(-VEL_MAX, min(VEL_MAX, vz))
    yaw_rate = max(-YAW_MAX, min(YAW_MAX, yaw_rate))

    # ------------------------------------------------------------------
    # Step 5 — Rotate world-frame xy velocity into the drone's body frame
    #
    # The inner loop (tello_controller.py) receives velocities relative
    # to the drone's heading (yaw), not the world axes.
    # We apply the inverse-yaw rotation to convert:
    #   vx_body =  vx·cos(yaw) + vy·sin(yaw)
    #   vy_body = -vx·sin(yaw) + vy·cos(yaw)
    # ------------------------------------------------------------------
    c, s    = math.cos(yaw), math.sin(yaw)
    vx_body =  vx * c + vy * s
    vy_body = -vx * s + vy * c

    return (vx_body, vy_body, vz, yaw_rate)
