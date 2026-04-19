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

# --- Derivative gains [x, y, z, yaw] ---
KD = [0.2, 0.2, 0.3, 1.0]

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

# Persistent state (survives between controller calls every dt seconds) 

# integral: Accumulate the error over time, which helps correct for any steady-state current_error in the system. 
integral  = [0.0, 0.0, 0.0, 0.0]
# last_target: Reset the integral whenever the target changes, preventing wind-up from previous targets
last_target = None
# last_error: Store the error from the previous time step for derivative calculation (if we were to implement a D term in the future). 
# Currently unused since we are only implementing PI control, but can be useful for debugging or future extensions.
last_error = [0.0, 0.0, 0.0, 0.0]  

def controller(state, target_pos, dt, wind_enabled=False):
    
    # state 
    # format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # Meaning: Current GPS location and orientation of the drone in the world frame. The drone's "forward" direction is along its yaw angle.

    # target_pos 
    # format: (x (m), y (m), z (m), yaw (radians))
    # Meaning: Desired position and yaw angle for the drone to reach. The drone should fly to this GPS location and orient itself to this yaw.

    # dt: time step (s)
    # Meaning: It means brain rethinks current situation every dt seconds. Used for integrating the error over time in the I term.

    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))

    # Global variables to store the integral of the error for each axis and the last target position.
    global integral, last_target, last_error

    # Unpack state and target
    x, y, z, _roll, _pitch, yaw = state
    tx, ty, tz, t_yaw = target_pos

    # Reset integral whenever the target changes
    if last_target is not None and tuple(target_pos) != last_target:
        integral = [0.0, 0.0, 0.0, 0.0]
    last_target = tuple(target_pos)

    # ------------------------------------------------------------------
    # Step 1 — Position current_error in the World Frame
    # ------------------------------------------------------------------
    
    # Compute the difference between the target and current position
    ex = tx - x
    ey = ty - y
    ez = tz - z
    # Wrap yaw error to [-π, π] so the drone always turns the short way
    e_yaw = (t_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

    current_error = [ex, ey, ez, e_yaw]

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
            integral[i] += current_error[i] * safe_dt
            integral[i]  = max(-INTEG_MAX, min(INTEG_MAX, integral[i]))

    # ------------------------------------------------------------------
    # Step 3 — PI control → world-frame velocity commands
    # ------------------------------------------------------------------
    
    # Compute the desired velocity in the world frame using the proportional, integral, and derivative terms of the PID controller. 
    # P term reacts to the current error
    # I term corrects for any accumulated error over time 
    # D term reacts to the rate of change of the error
    vx = KP[0] * ex + KI[0] * integral[0] + KD[0] * ((current_error[0] - last_error[0]) / safe_dt)
    vy = KP[1] * ey + KI[1] * integral[1] + KD[1] * ((current_error[1] - last_error[1]) / safe_dt)
    vz = KP[2] * ez + KI[2] * integral[2] + KD[2] * ((current_error[2] - last_error[2]) / safe_dt)
    yaw_rate = KP[3] * e_yaw + KI[3] * integral[3] + KD[3] * ((current_error[3] - last_error[3]) / safe_dt)

    # ------------------------------------------------------------------
    # Step 4 — Clamp to simulator speed limits
    # ------------------------------------------------------------------
    vx = max(-VEL_MAX, min(VEL_MAX, vx))
    vy = max(-VEL_MAX, min(VEL_MAX, vy))
    vz = max(-VEL_MAX, min(VEL_MAX, vz))
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

    # vx_body
    # How fast to move forward/backward relative to the drone's current facing direction. Positive vx_body means move forward, negative means move backward.
    vx_body =  vx * c + vy * s

    # vy_body
    # How fast to move left/right relative to the drone's current facing direction. Positive vy_body means move to the right, negative means move to the left.
    vy_body = -vx * s + vy * c

    # vz is already in the body frame (up/down), and yaw_rate is unaffected by rotation, so we keep them as is.
    # yaw_rate is how fast to rotate around the vertical axis. Positive yaw_rate means rotate clockwise, negative means rotate counterclockwise.
    output = (vx_body, vy_body, vz, yaw_rate)

    # Store the current error for potential future use (e.g., if we were to implement a D term)
    last_error = current_error  

    return output