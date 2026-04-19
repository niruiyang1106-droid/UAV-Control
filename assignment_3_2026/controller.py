#wind_enabled = True   # Set True → simulator applies wind disturbances during marking

import math

# =====================================================================
# CASCADE PID POSITION CONTROLLER  (runs at 20 Hz via run.py)
#
# Advanced method: 3-layer Cascade PID controller.
#
#   Layer 1 (Outer — Position PID):
#       Input:  position error = target_pos - current_pos
#       Output: desired velocity setpoint (world frame)
#       This outer loop tells the drone how fast to move in each axis
#       to close the gap between current and target position.
#
#   Layer 2 (Middle — Velocity PI):
#       Input:  velocity error = desired_velocity - estimated_velocity
#       Output: refined velocity correction
#       Estimated velocity is computed from position differences (Δpos / Δt).
#       This middle loop corrects any mismatch between the commanded
#       velocity and what the drone is actually achieving.
#
#   Layer 3 (Inner — handled by tello_controller.py):
#       Input:  final velocity command from Layer 2
#       Output: attitude → rate → motor RPMs
#       This hardware-level cascade executes the velocity commands.
#
# For each axis we use a PID (or PI) controller:
#   P (Proportional): react immediately to the current error
#   I (Integral):     slowly correct any small residual offset (e.g. from wind)
#   D (Derivative):   damp oscillations by reacting to error rate of change
#
# The output is a 4-element tuple: (vx, vy, vz, yaw_rate)
# expressed in the drone's yaw-body frame (not world frame).
# =====================================================================

# -----------------------------------------------------------------------
# Layer 1 gains: Position PID [x, y, z, yaw]
# Convert position error (metres) into a desired velocity setpoint (m/s).
# -----------------------------------------------------------------------
# Increase KP to fly toward the target faster and hold more tightly.
# Too large → oscillation.
POS_KP = [0.85, 0.85, 1.4, 3.0]
POS_KI = [0.12, 0.12, 0.15, 0.1]
POS_KD = [0.2,  0.2,  0.2,  0.05]

# -----------------------------------------------------------------------
# Layer 2 gains: Velocity P [x, y, z]
# Correct the error between desired and estimated actual velocity.
# Yaw rate is already well-controlled by Layer 1 alone, so only x, y, z.
#
# NOTE: VEL_KI is intentionally set to zero.
# Velocity is estimated from noisy position differences (Δpos/Δt).
# Integrating that noisy signal causes the integral to drift (random walk)
# until it hits the anti-windup limit, which then injects a spurious
# constant offset into the velocity command and causes oscillation.
# A pure P correction is enough to reduce velocity tracking error.
# -----------------------------------------------------------------------
VEL_KP = [0.1,  0.1,  0.1]
VEL_KI = [0.0,  0.0,  0.0]

# --- Speed limits (must match simulator's ±1 m/s / ±1.745 rad/s clip) ---
VEL_MAX = 1.0    # m/s
YAW_MAX = 1.745  # rad/s

# --- Anti-windup clamp ---
# Prevents the integral from growing unboundedly if the drone is stuck.
POS_INTEG_MAX = 2.0   # max integral magnitude for position loop  (m·s)
VEL_INTEG_MAX = 0.3   # max integral magnitude for velocity loop  (m·s)

# --- Filter coefficients for derivative and velocity estimation ---
# Higher value = smoother but slightly slower response. Range: 0 → 1.
DERIV_ALPHA = 0.5   # Position error derivative filter (heavy — avoids noisy D kicks)
VEL_ALPHA   = 0.5    # Velocity estimation filter (very heavy — Δpos/Δt is inherently noisy)

# -----------------------------------------------------------------------
# Persistent state (survives between controller calls every dt seconds)
# -----------------------------------------------------------------------

# pos_integral: Accumulate the position error over time for the I term of Layer 1.
# Helps correct for any steady-state offset in the system (e.g. constant wind).
pos_integral = [0.0, 0.0, 0.0, 0.0]

# vel_integral: Accumulate the velocity error over time for the I term of Layer 2.
vel_integral = [0.0, 0.0, 0.0]

# last_target: Reset the integrals whenever the target changes,
# preventing wind-up from a previous target.
last_target = None

# last_pos: Store the previous position so we can estimate the drone's velocity
# using the backward-difference formula: estimated_velocity = (Δpos / Δt).
last_pos = None

# last_pos_err: Store the previous position error for the D term.
# Set to None on the first call so we can safely skip the D term on initialisation.
last_pos_err = None

# pos_d_filtered: Low-pass filtered derivative of the position error (reduces noise).
pos_d_filtered = [0.0, 0.0, 0.0, 0.0]

# est_vel: Estimated current velocity of the drone in the world frame (filtered).
est_vel = [0.0, 0.0, 0.0]


def controller(state, target_pos, dt, wind_enabled=False):

    # state
    # format: [position_x (m), position_y (m), position_z (m), roll (rad), pitch (rad), yaw (rad)]
    # Meaning: Current GPS location and orientation of the drone in the world frame.
    # The drone's "forward" direction is along its yaw angle.

    # target_pos
    # format: (x (m), y (m), z (m), yaw (rad))
    # Meaning: Desired position and yaw angle for the drone to reach.

    # dt: time step (s)
    # Meaning: The controller is called every dt seconds.  Used to integrate error over time.

    # return velocity command format:
    # (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s),
    #  velocity_z_setpoint (m/s), yaw_rate_setpoint (rad/s))

    # Global variables to persist state between calls
    global pos_integral, vel_integral, last_target, last_pos
    global last_pos_err, pos_d_filtered, est_vel

    # Unpack current state and target
    x, y, z, _roll, _pitch, yaw = state
    tx, ty, tz, t_yaw = target_pos
    safe_dt = max(dt, 1e-6)   # Avoid division by zero

    # Reset all integrals and derivative state whenever the target changes.
    # Without this, a sudden target jump would create a large one-frame derivative
    # spike (the "derivative kick") that can jolt the drone violently.
    if last_target is not None and tuple(target_pos) != last_target:
        pos_integral    = [0.0, 0.0, 0.0, 0.0]
        vel_integral    = [0.0, 0.0, 0.0]
        pos_d_filtered  = [0.0, 0.0, 0.0, 0.0]
        est_vel         = [0.0, 0.0, 0.0]
        last_pos_err    = None   # Force skip of D term on the very next call
    last_target = tuple(target_pos)

    # =======================================================================
    # LAYER 1: Position PID
    # Compute position error and calculate the desired velocity setpoint.
    # =======================================================================

    # ------------------------------------------------------------------
    # Step 1 — Position error in the World Frame
    # ------------------------------------------------------------------
    ex    = tx - x
    ey    = ty - y
    ez    = tz - z
    # Wrap yaw error to [-π, π] so the drone always turns the short way
    e_yaw = (t_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

    pos_err = [ex, ey, ez, e_yaw]

    # ------------------------------------------------------------------
    # Step 2 — Update the position integrals (per-axis saturation anti-windup)
    #
    # Each axis integrates independently only when its own P term is unsaturated.
    # During a long-distance approach the P term hits the velocity limit, so
    # accumulating the integral would cause overshoot on arrival.
    # Once close enough (P term below VEL_MAX), the integral builds and
    # eliminates any residual steady-state error (e.g. caused by wind).
    #
    # Using per-axis checks (not a combined OR) so that, for example, altitude
    # can still integrate even while the drone is far away horizontally.
    # Yaw always integrates because it has no velocity-saturation issue.
    # ------------------------------------------------------------------
    # x axis: integrate only when the x P-output would not saturate
    if abs(POS_KP[0] * ex) < VEL_MAX:
        pos_integral[0] += pos_err[0] * safe_dt
        pos_integral[0]  = max(-POS_INTEG_MAX, min(POS_INTEG_MAX, pos_integral[0]))

    # y axis
    if abs(POS_KP[1] * ey) < VEL_MAX:
        pos_integral[1] += pos_err[1] * safe_dt
        pos_integral[1]  = max(-POS_INTEG_MAX, min(POS_INTEG_MAX, pos_integral[1]))

    # z axis
    if abs(POS_KP[2] * ez) < VEL_MAX:
        pos_integral[2] += pos_err[2] * safe_dt
        pos_integral[2]  = max(-POS_INTEG_MAX, min(POS_INTEG_MAX, pos_integral[2]))

    # Yaw integral — always active
    pos_integral[3] += pos_err[3] * safe_dt
    pos_integral[3]  = max(-POS_INTEG_MAX, min(POS_INTEG_MAX, pos_integral[3]))

    # ------------------------------------------------------------------
    # Step 3 — Compute filtered derivative of the position error
    #
    # D term reacts to the rate of change of the error, damping oscillations.
    # On the very first call there is no previous error, so we skip D.
    # The low-pass filter (DERIV_ALPHA) smooths sensor noise in the derivative.
    # ------------------------------------------------------------------
    if last_pos_err is None:
        # First call: no previous error available — initialise derivative to zero
        pos_d_filtered = [0.0, 0.0, 0.0, 0.0]
    else:
        for i in range(4):
            raw_d = (pos_err[i] - last_pos_err[i]) / safe_dt
            # Exponential moving-average filter: mix old and new derivative values
            pos_d_filtered[i] = (DERIV_ALPHA * pos_d_filtered[i] +
                                 (1.0 - DERIV_ALPHA) * raw_d)

    # Save the current position error ready for the next call's D calculation
    last_pos_err = pos_err[:]

    # ------------------------------------------------------------------
    # Step 4 — PID calculation → desired velocity in the world frame
    #
    # Compute the desired velocity in the world frame using the proportional,
    # integral, and derivative terms of the PID controller.
    # P term reacts to the current error
    # I term corrects for any accumulated error over time
    # D term reacts to the rate of change of the error
    # ------------------------------------------------------------------
    vx_des   = (POS_KP[0] * ex    + POS_KI[0] * pos_integral[0]
                                   + POS_KD[0] * pos_d_filtered[0])
    vy_des   = (POS_KP[1] * ey    + POS_KI[1] * pos_integral[1]
                                   + POS_KD[1] * pos_d_filtered[1])
    vz_des   = (POS_KP[2] * ez    + POS_KI[2] * pos_integral[2]
                                   + POS_KD[2] * pos_d_filtered[2])
    yaw_rate = (POS_KP[3] * e_yaw + POS_KI[3] * pos_integral[3]
                                   + POS_KD[3] * pos_d_filtered[3])

    # ------------------------------------------------------------------
    # Step 5 — Clamp desired velocity to simulator speed limits
    # ------------------------------------------------------------------
    vx_des   = max(-VEL_MAX, min(VEL_MAX, vx_des))
    vy_des   = max(-VEL_MAX, min(VEL_MAX, vy_des))
    vz_des   = max(-VEL_MAX, min(VEL_MAX, vz_des))
    yaw_rate = max(-YAW_MAX, min(YAW_MAX, yaw_rate))

    # =======================================================================
    # LAYER 2: Velocity PI (cascade refinement)
    #
    # Estimate the drone's actual velocity from the change in position, then
    # correct the velocity command based on the velocity tracking error.
    # This middle loop improves accuracy by ensuring the velocity setpoint
    # from Layer 1 is actually being achieved by the drone.
    # =======================================================================

    # Estimate current velocity using the backward difference method (Δpos / Δt)
    if last_pos is not None:
        raw_vx_est = (x - last_pos[0]) / safe_dt
        raw_vy_est = (y - last_pos[1]) / safe_dt
        raw_vz_est = (z - last_pos[2]) / safe_dt
        # Apply a very heavy low-pass filter (VEL_ALPHA=0.9) to smooth the noisy
        # velocity estimate.  Δpos/Δt at 20 Hz amplifies even tiny position noise.
        est_vel[0] = VEL_ALPHA * est_vel[0] + (1.0 - VEL_ALPHA) * raw_vx_est
        est_vel[1] = VEL_ALPHA * est_vel[1] + (1.0 - VEL_ALPHA) * raw_vy_est
        est_vel[2] = VEL_ALPHA * est_vel[2] + (1.0 - VEL_ALPHA) * raw_vz_est

    # Save the current position so the next call can estimate velocity
    last_pos = [x, y, z]

    # Velocity error: how far the estimated actual velocity is from the desired
    vel_err_x = vx_des - est_vel[0]
    vel_err_y = vy_des - est_vel[1]
    vel_err_z = vz_des - est_vel[2]

    # Update velocity integrals with anti-windup clamp
    vel_integral[0] += vel_err_x * safe_dt
    vel_integral[1] += vel_err_y * safe_dt
    vel_integral[2] += vel_err_z * safe_dt
    for i in range(3):
        vel_integral[i] = max(-VEL_INTEG_MAX, min(VEL_INTEG_MAX, vel_integral[i]))

    # PI output: velocity correction added on top of the Layer 1 setpoint
    vx_corr = VEL_KP[0] * vel_err_x + VEL_KI[0] * vel_integral[0]
    vy_corr = VEL_KP[1] * vel_err_y + VEL_KI[1] * vel_integral[1]
    vz_corr = VEL_KP[2] * vel_err_z + VEL_KI[2] * vel_integral[2]

    # Combine Layer 1 setpoint and Layer 2 correction, then clamp to limits
    vx = max(-VEL_MAX, min(VEL_MAX, vx_des + vx_corr))
    vy = max(-VEL_MAX, min(VEL_MAX, vy_des + vy_corr))
    vz = max(-VEL_MAX, min(VEL_MAX, vz_des + vz_corr))

    # =======================================================================
    # Step 6 — Rotate world-frame xy velocity into the drone's body frame
    #
    # The inner loop (tello_controller.py) receives velocities relative
    # to the drone's heading (yaw), not the world axes.
    # We apply the inverse-yaw rotation to convert:
    #   vx_body =  vx·cos(yaw) + vy·sin(yaw)
    #   vy_body = -vx·sin(yaw) + vy·cos(yaw)
    # =======================================================================
    c = math.cos(yaw)
    s = math.sin(yaw)

    # vx_body
    # How fast to move forward/backward relative to the drone's current facing direction.
    # Positive vx_body means move forward, negative means move backward.
    vx_body =  vx * c + vy * s

    # vy_body
    # How fast to move left/right relative to the drone's current facing direction.
    # Positive vy_body means move to the right, negative means move to the left.
    vy_body = -vx * s + vy * c

    # vz is already in the body frame (up/down), and yaw_rate is unaffected by rotation,
    # so we keep them as is.
    # yaw_rate is how fast to rotate around the vertical axis.
    # Positive yaw_rate means rotate clockwise, negative means rotate counterclockwise.
    output = (vx_body, vy_body, vz, yaw_rate)

    return output
