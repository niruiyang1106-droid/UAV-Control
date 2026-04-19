#wind_enabled = True   # Set True → simulator applies wind disturbances during marking

import math

# =====================================================================
# SINGLE-LOOP POSITION PID + DOBC (Disturbance Observer-Based Control)
#
# Advanced method: Disturbance Observer-Based Controller (DOBC)
# superimposed on a classic single-loop Position PID.
#
# Architecture overview:
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │  target_pos ──► [PID] ──► vx_pid, vy_pid (World Frame)       │
#   │                    │                                         │
#   │  DOBC:             ▼                                         │
#   │  vel_est ─► disturbance ─► [EMA filter] ─► wind_est          │
#   │                                                  │           │
#   │  vx_final = vx_pid − wind_est_x  ◄──────────────┘            │
#   │  vy_final = vy_pid − wind_est_y                              │
#   │                    │                                         │
#   │  [Yaw rotation] ──► vx_body, vy_body (Body Frame)  ──► UAV   │
#   └──────────────────────────────────────────────────────────────┘
#
# Single-Loop PID:
#   Converts position error directly to a velocity command in the
#   world (global) frame for all four axes (x, y, z, yaw).
#   No inner velocity-tracking loop — keeps the design simple and
#   reduces the total number of gains that need tuning.
#
# DOBC (X and Y only — wind is predominantly horizontal):
#   Estimates the steady-state wind disturbance by comparing the
#   drone's observed velocity (differentiated from position) against
#   the last velocity command we issued.  A heavy exponential moving-
#   average filter (α = 0.98) extracts only the slow, persistent
#   component, ignoring transient motion and sensor noise.  The
#   estimate is then fed forward as a correction to the PID output
#   so the drone automatically compensates for constant wind without
#   waiting for the integral term to wind up.
#
# Coordinate frames:
#   • All PID calculations and DOBC run in the World (global) Frame.
#   • Only at the very last step are vx/vy rotated into the drone's
#     Body Frame via the inverse-yaw rotation matrix.
# =====================================================================

# -----------------------------------------------------------------------
# Position PID gains  [x, y, z, yaw]
# P — how aggressively to chase position error
# I — slow correction for residual offset (wind primarily handled by DOBC)
# D — damp oscillations; operates on a filtered derivative to reduce noise
# -----------------------------------------------------------------------
POS_KP = [0.65, 0.65, 1.2,  2.8 ]
POS_KI = [0.05, 0.05, 0.10, 0.07]
POS_KD = [0.60, 0.60, 0.22, 0.05]

# -----------------------------------------------------------------------
# Speed limits (must match the simulator's internal clip values)
# -----------------------------------------------------------------------
VEL_MAX = 1.0    # m/s  — maximum translational velocity per axis
YAW_MAX = 1.745  # rad/s — maximum yaw rate

# -----------------------------------------------------------------------
# Anti-windup: conditional integration
# The integral is only allowed to accumulate when the drone is already
# close to the target.  During a long-range approach the proportional
# term is saturated at VEL_MAX, so integrating would cause overshoot.
# -----------------------------------------------------------------------
INTEG_DIST_THRESH = 0.6   # metres  — activate integral only when already close
INTEG_YAW_THRESH  = 0.3   # radians — yaw error gate for yaw integral
POS_INTEG_MAX     = 1.2   # maximum integral magnitude (m·s)

# -----------------------------------------------------------------------
# Derivative filter (exponential moving average)
# Smooths the raw Δerror/Δt signal before using it as the D term.
# Higher α → smoother but slower derivative response.
# -----------------------------------------------------------------------
DERIV_ALPHA = 0.65   # heavier filter; KD raised above to compensate for the attenuation

# -----------------------------------------------------------------------
# DOBC parameters
# -----------------------------------------------------------------------
# DOBC_ALPHA: EMA coefficient for the wind estimator.
#   0.98 means the filter has a time constant of roughly dt/(1−0.98) ≈ 2.5 s
#   (at 20 Hz).  This ensures only a truly persistent bias (steady wind)
#   survives while transient accelerations and measurement noise are rejected.
DOBC_ALPHA = 0.98

# DOBC_GAIN: how much of the wind estimate to subtract.
#   0.8 < 1.0 adds a small safety margin against velocity-estimate noise
#   near the setpoint where last_cmd ≈ 0 and disturbance calc is noisiest.
DOBC_GAIN = 0.8

# DOBC_MAX: hard cap on the wind compensation magnitude (m/s per axis).
#   Prevents a corrupted wind_est from overwhelming the PID command.
DOBC_MAX = 0.25

# VEL_EST_ALPHA: pre-filter for the raw Δpos/Δt velocity estimate.
#   Heavier filter (0.7) reduces quantisation noise before DOBC disturbance calc.
VEL_EST_ALPHA = 0.7

# -----------------------------------------------------------------------
# Persistent state — survives across controller calls
# -----------------------------------------------------------------------
pos_integral   = [0.0, 0.0, 0.0, 0.0]  # Accumulated position error [x,y,z,yaw]
last_target    = None                    # Detect target changes → reset integrals
last_pos       = None                    # Previous position for Δpos/Δt
last_pos_err   = None                    # Previous pos error for D term
pos_d_filtered = [0.0, 0.0, 0.0, 0.0]  # Low-pass filtered derivative

# DOBC state (world frame, x and y only)
vel_est  = [0.0, 0.0]   # Smoothed velocity estimate  [vx, vy]
wind_est = [0.0, 0.0]   # Smoothed wind disturbance   [wx, wy]
last_cmd = [0.0, 0.0]   # Last world-frame PID command [vx, vy]


def controller(state, target_pos, dt, wind_enabled=False):
    """
    Single-loop Position PID with DOBC feedforward wind compensation.

    Inputs
    ------
    state      : [x, y, z, roll, pitch, yaw]  — world frame (m, rad)
    target_pos : (tx, ty, tz, t_yaw)          — desired state (m, rad)
    dt         : control loop period (s)
    wind_enabled: passed by simulator (unused inside the function)

    Returns
    -------
    (vx_body, vy_body, vz, yaw_rate) — velocity commands in body frame (m/s, rad/s)
    """

    global pos_integral, last_target, last_pos
    global last_pos_err, pos_d_filtered
    global vel_est, wind_est, last_cmd

    # Unpack state and target
    x, y, z, _roll, _pitch, yaw = state
    tx, ty, tz, t_yaw = target_pos
    safe_dt = max(dt, 1e-6)   # Guard against dt = 0 at startup

    # -------------------------------------------------------------------
    # Housekeeping: reset integral / derivative state on target change.
    # A sudden target jump produces a large one-step derivative spike
    # (the "derivative kick") — clearing last_pos_err avoids that.
    # The wind estimate is deliberately NOT reset: wind persists across
    # target changes and the observer should keep its learned value.
    # -------------------------------------------------------------------
    if last_target is not None and tuple(target_pos) != last_target:
        pos_integral   = [0.0, 0.0, 0.0, 0.0]
        pos_d_filtered = [0.0, 0.0, 0.0, 0.0]
        last_pos_err   = None   # Skip D term on the very next call
    last_target = tuple(target_pos)

    # ===================================================================
    # STEP 1 — Position error in the World Frame
    # ===================================================================
    ex    = tx - x
    ey    = ty - y
    ez    = tz - z
    # Wrap yaw error to (−π, π] so the drone always turns the short way
    e_yaw = (t_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi

    pos_err = [ex, ey, ez, e_yaw]

    # ===================================================================
    # STEP 2 — Conditional Integration (Anti-windup)
    #
    # Accumulate the integral only when close to the target.
    # Far away, the P term already saturates at VEL_MAX; integrating
    # on top of a saturated P term causes overshoot.
    # x/y/z share a 3-D distance gate; yaw has a separate angular gate.
    # ===================================================================
    dist_3d = math.sqrt(ex*ex + ey*ey + ez*ez)

    if dist_3d < INTEG_DIST_THRESH:
        for i in range(3):                   # x, y, z
            pos_integral[i] += pos_err[i] * safe_dt
            pos_integral[i]  = max(-POS_INTEG_MAX,
                                    min(POS_INTEG_MAX, pos_integral[i]))

    if abs(e_yaw) < INTEG_YAW_THRESH:       # yaw
        pos_integral[3] += pos_err[3] * safe_dt
        pos_integral[3]  = max(-POS_INTEG_MAX,
                                min(POS_INTEG_MAX, pos_integral[3]))

    # ===================================================================
    # STEP 3 — Filtered Derivative of Position Error
    #
    # D term damps oscillations by reacting to how fast the error is
    # shrinking.  The raw finite difference is noisy, so we apply an EMA.
    # On the first call last_pos_err is None — we skip D to avoid a spike.
    # ===================================================================
    if last_pos_err is None:
        pos_d_filtered = [0.0, 0.0, 0.0, 0.0]
    else:
        for i in range(4):
            raw_d = (pos_err[i] - last_pos_err[i]) / safe_dt
            # EMA: blend previous filtered value with new raw derivative
            pos_d_filtered[i] = (DERIV_ALPHA * pos_d_filtered[i]
                                 + (1.0 - DERIV_ALPHA) * raw_d)

    last_pos_err = pos_err[:]   # Save for next call

    # ===================================================================
    # STEP 4 — Single-Loop Position PID → desired velocity (World Frame)
    #
    #   v_cmd = Kp · e  +  Ki · ∫e dt  +  Kd · ė_filtered
    # ===================================================================
    vx_pid = (POS_KP[0] * ex
              + POS_KI[0] * pos_integral[0]
              + POS_KD[0] * pos_d_filtered[0])

    vy_pid = (POS_KP[1] * ey
              + POS_KI[1] * pos_integral[1]
              + POS_KD[1] * pos_d_filtered[1])

    vz_pid = (POS_KP[2] * ez
              + POS_KI[2] * pos_integral[2]
              + POS_KD[2] * pos_d_filtered[2])

    yaw_rate_pid = (POS_KP[3] * e_yaw
                    + POS_KI[3] * pos_integral[3]
                    + POS_KD[3] * pos_d_filtered[3])

    # Clamp to simulator speed limits before DOBC
    vx_pid       = max(-VEL_MAX, min(VEL_MAX, vx_pid))
    vy_pid       = max(-VEL_MAX, min(VEL_MAX, vy_pid))
    vz_pid       = max(-VEL_MAX, min(VEL_MAX, vz_pid))
    yaw_rate_pid = max(-YAW_MAX, min(YAW_MAX, yaw_rate_pid))

    # ===================================================================
    # STEP 5 — DOBC: Disturbance Observer-Based Feedforward Compensation
    #          (X and Y world-frame axes only; wind assumed horizontal)
    #
    # Theory:
    #   In the absence of disturbance:  actual_vel ≈ last_cmd
    #   With steady wind:               actual_vel = last_cmd + wind
    #   Therefore:                      wind ≈ actual_vel − last_cmd
    #
    # Implementation:
    #   a) Velocity differentiation — estimate actual velocity from Δpos/Δt
    #   b) Light pre-filter (VEL_EST_ALPHA) to smooth quantisation noise
    #   c) Disturbance = smoothed velocity estimate − last issued command
    #   d) Heavy EMA (DOBC_ALPHA = 0.98) keeps only the DC/steady component
    #   e) Feedforward: subtract wind estimate from PID output
    # ===================================================================
    if last_pos is not None:
        # (a) Velocity differentiation in the world frame
        raw_vx_est = (x - last_pos[0]) / safe_dt
        raw_vy_est = (y - last_pos[1]) / safe_dt

        # (b) Light EMA pre-filter to reduce Δpos/Δt noise
        vel_est[0] = VEL_EST_ALPHA * vel_est[0] + (1.0 - VEL_EST_ALPHA) * raw_vx_est
        vel_est[1] = VEL_EST_ALPHA * vel_est[1] + (1.0 - VEL_EST_ALPHA) * raw_vy_est

        # (c) Disturbance = observed velocity − last commanded velocity
        #   Positive: drone moved faster than commanded → tailwind in that axis
        #   Negative: drone moved slower than commanded → headwind in that axis
        disturbance_x = vel_est[0] - last_cmd[0]
        disturbance_y = vel_est[1] - last_cmd[1]

        # (d) Heavy EMA (α = 0.98): time constant ≈ dt/(1−α) ≈ 2.5 s at 20 Hz.
        #   Only a truly persistent (steady-state) bias survives the filter.
        #   Transient accelerations and noise are averaged away.
        wind_est[0] = DOBC_ALPHA * wind_est[0] + (1.0 - DOBC_ALPHA) * disturbance_x
        wind_est[1] = DOBC_ALPHA * wind_est[1] + (1.0 - DOBC_ALPHA) * disturbance_y

    # Store current position for next call
    last_pos = [x, y, z]

    # (e) Feedforward compensation: subtract the wind estimate.
    #   Clamp the correction to ±DOBC_MAX so a noisy wind_est (which can
    #   occur near the setpoint where last_cmd ≈ 0) cannot overpower the PID.
    comp_x = max(-DOBC_MAX, min(DOBC_MAX, DOBC_GAIN * wind_est[0]))
    comp_y = max(-DOBC_MAX, min(DOBC_MAX, DOBC_GAIN * wind_est[1]))
    vx_final = vx_pid - comp_x
    vy_final = vy_pid - comp_y
    vz_final       = vz_pid         # No DOBC on Z (vertical wind negligible)
    yaw_rate_final = yaw_rate_pid   # No DOBC on yaw

    # Re-clamp after DOBC addition (compensation can push slightly over limit)
    vx_final = max(-VEL_MAX, min(VEL_MAX, vx_final))
    vy_final = max(-VEL_MAX, min(VEL_MAX, vy_final))

    # Save the world-frame command (pre-rotation) so the next DOBC step
    # can compute the disturbance correctly in the same coordinate frame.
    last_cmd[0] = vx_final
    last_cmd[1] = vy_final

    # ===================================================================
    # STEP 6 — World Frame → Body Frame rotation
    #
    # The simulator expects velocity commands expressed relative to the
    # drone's current heading (yaw), not the global axes.
    # Apply the inverse-yaw (transpose) rotation matrix:
    #
    #   [ vx_body ]   [  cos(yaw)  sin(yaw) ] [ vx_world ]
    #   [ vy_body ] = [ −sin(yaw)  cos(yaw) ] [ vy_world ]
    #
    # vz and yaw_rate are invariant under this rotation (vertical axis).
    # ===================================================================
    c = math.cos(yaw)
    s = math.sin(yaw)

    vx_body =  vx_final * c + vy_final * s
    vy_body = -vx_final * s + vy_final * c

    return (vx_body, vy_body, vz_final, yaw_rate_final)
