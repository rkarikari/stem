import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="RadioSport Orbital.X",
    page_icon="🧟",
    layout="centered",
    menu_items={
        'Report a Bug': "https://github.com/rkarikari/RadioSport-chat",
        'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
    }
)

# Constants for transfer efficiency
G, M = 1.0, 1.0
DEFAULT_DT = 0.08
APPROACH_DT = 0.009
STEPS = 11000
APPROACH_THRESHOLD = 0.7
ARRIVAL_THRESHOLD = 0.050
SUN_DANGER_RADIUS = 1.1
EARTH_TO_MARS_LAUNCH_ADJUSTMENT = -0.198
MARS_TO_EARTH_LAUNCH_ADJUSTMENT = 0.235
STRICT_LAUNCH_WINDOW_DEGREES = 1.5
STRICT_LAUNCH_WINDOW = np.deg2rad(STRICT_LAUNCH_WINDOW_DEGREES)

# Time conversion factor (time units to months)
CONVERSION_FACTOR = 8.5 / (np.pi * np.sqrt((2.0 + 3.8) / 2)**3)  # Approx 0.5473 months per time unit

# Orbital setup
r_mars = np.array([3.8, 0.0])
v_mars = np.array([0.0, np.sqrt(G * M / 3.8)])
earth_angle = -np.pi / 1.08
r_earth = np.array([2.0 * np.cos(earth_angle), 2.0 * np.sin(earth_angle)])
v_earth = np.array([-np.sin(earth_angle), np.cos(earth_angle)]) * np.sqrt(G * M / 2.0)

# Transfer calculations
r_e, r_m = 2.0, 3.8
a_transfer_hohmann = (r_e + r_m) / 2
hohmann_transfer_time = np.pi * np.sqrt(a_transfer_hohmann**3 / (G * M)) * CONVERSION_FACTOR
synodic_period = 2 * np.pi / abs(np.sqrt(G * M / r_e**3) - np.sqrt(G * M / r_m**3)) * CONVERSION_FACTOR
omega_e = np.sqrt(G * M / r_e**3)
omega_m = np.sqrt(G * M / r_m**3)

# Sidebar setup
st.title("🚀 Orbital X - GrokX")
sidebar_mission_status = st.sidebar.empty()
sidebar_launch_window = st.sidebar.empty()
sidebar_trajectory = st.sidebar.empty()
sidebar_performance = st.sidebar.empty()
sidebar_proximity = st.sidebar.empty()
sidebar_safety = st.sidebar.empty()
sidebar_efficiency = st.sidebar.empty()
sidebar_completion = st.sidebar.empty()

# State variables
state = "on_earth"
r_ship, v_ship = r_earth.copy(), v_earth.copy()
mission_timer = 0
successful_missions = 0
mars_visits = 0
total_launches = 0
min_sun_distance = float('inf')
approach_phase = "none"
landing_speed = 0.0
fuel_used = 0.0
launch_attempt_timer = 0
last_launch_time = 0
launch_wait_time = 1.5
min_proximity_to_target = float('inf')
proximity_history = []
current_target = "none"
transfer_start_time = 0
current_transfer_time = 0
mission_start_times = []
mission_durations = []
launch_window_efficiency = 0.0
trajectory_accuracy = 100.0
max_velocity = 0.0
total_delta_v = 0.0
sun_safety_violations = 0
orbital_energy = 0.0
current_dt = DEFAULT_DT

def acceleration(r):
    r_norm = np.linalg.norm(r)
    return -G * M * r / r_norm**3 if r_norm > 0.005 else np.zeros(2)

def hohmann_velocity(from_radius, to_radius, current_pos, current_vel):
    a = (from_radius + to_radius) / 2
    v_transfer_mag = np.sqrt(G * M * (2 / from_radius - 1 / a))
    position_norm = np.linalg.norm(current_pos)
    tangent_direction = np.array([-current_pos[1], current_pos[0]]) / position_norm
    v_transfer = v_transfer_mag * tangent_direction
    radial_direction = current_pos / position_norm
    v_transfer += radial_direction * (v_transfer_mag * 0.07)
    delta_v = np.linalg.norm(v_transfer - current_vel)
    global fuel_used, total_delta_v
    fuel_used += delta_v
    total_delta_v += delta_v
    return v_transfer

def adaptive_approach_burn(ship_pos, ship_vel, target_pos, target_vel, target_radius, phase):
    relative_pos = ship_pos - target_pos
    relative_vel = ship_vel - target_vel
    distance = np.linalg.norm(relative_pos)
    v_target_circular = np.sqrt(G * M / target_radius)
    correction = np.zeros(2)
    
    if phase == "far_approach" and distance < 1.2:
        desired_speed = 0.12
        approach_direction = -relative_pos / distance if distance > 0 else np.zeros(2)
        desired_vel = target_vel + desired_speed * approach_direction
        correction = (desired_vel - ship_vel) * 0.6
    elif phase == "close_approach" and distance < 0.45:
        desired_speed = 0.05
        approach_direction = -relative_pos / distance if distance > 0 else np.zeros(2)
        desired_vel = target_vel + desired_speed * approach_direction
        correction = (desired_vel - ship_vel) * 1.2
    elif phase == "final_capture" and distance < 0.20:
        target_tangent = np.array([-target_pos[1], target_pos[0]]) / np.linalg.norm(target_pos) if np.linalg.norm(target_pos) > 0 else np.zeros(2)
        desired_vel = v_target_circular * target_tangent
        position_correction = -relative_pos * 0.25
        velocity_correction = (desired_vel - ship_vel) * 1.5
        correction = velocity_correction + position_correction
    
    if np.linalg.norm(correction) > 0:
        correction_magnitude = np.linalg.norm(correction) * current_dt * 0.12
        global fuel_used, total_delta_v
        fuel_used += correction_magnitude
        total_delta_v += correction_magnitude
    return correction

def calculate_relative_velocity(ship_vel, target_vel):
    return np.linalg.norm(ship_vel - target_vel)

def normalize_angle(angle):
    return angle % (2 * np.pi)

def optimal_launch_window(r_earth_current, r_mars_current, v_earth_current, v_mars_current, destination):
    theta_earth = normalize_angle(np.arctan2(r_earth_current[1], r_earth_current[0]))
    theta_mars = normalize_angle(np.arctan2(r_mars_current[1], r_mars_current[0]))
    
    if destination == "mars":
        mars_movement_hohmann = omega_m * hohmann_transfer_time / CONVERSION_FACTOR
        target_lead_angle = np.pi - mars_movement_hohmann + EARTH_TO_MARS_LAUNCH_ADJUSTMENT
        current_phase = normalize_angle(theta_mars - theta_earth)
        phase_error = current_phase - target_lead_angle
        if phase_error < -np.pi: phase_error += 2 * np.pi
        elif phase_error > np.pi: phase_error -= 2 * np.pi
        optimal_window_size = 0.25
        window_open = abs(phase_error) < optimal_window_size and phase_error <= 0
        efficiency_rating = max(0, 1.0 - (abs(phase_error) / optimal_window_size))
    else:
        earth_movement_hohmann = omega_e * hohmann_transfer_time / CONVERSION_FACTOR
        current_separation = normalize_angle(theta_earth - theta_mars)
        target_separation = np.pi - earth_movement_hohmann + MARS_TO_EARTH_LAUNCH_ADJUSTMENT
        target_separation = normalize_angle(target_separation)
        phase_error = current_separation - target_separation
        if phase_error > np.pi: phase_error -= 2 * np.pi
        elif phase_error < -np.pi: phase_error += 2 * np.pi
        optimal_window_size = 0.55
        window_open = abs(phase_error) < optimal_window_size
        efficiency_rating = max(0, 1.0 - (abs(phase_error) / optimal_window_size))
    
    return window_open, phase_error, efficiency_rating

def update_proximity_tracking(ship_pos, target_pos, target_name):
    global min_proximity_to_target, proximity_history, current_target
    distance = np.linalg.norm(ship_pos - target_pos)
    if current_target != target_name:
        min_proximity_to_target = float('inf')
        proximity_history = []
        current_target = target_name
    min_proximity_to_target = min(min_proximity_to_target, distance)
    proximity_history.append(distance)
    if len(proximity_history) > 50: proximity_history.pop(0)

def calculate_orbital_energy(pos, vel):
    kinetic = 0.5 * np.linalg.norm(vel)**2
    potential = -G * M / np.linalg.norm(pos)
    return kinetic + potential

# Visualization setup
fig, ax1 = plt.subplots(1, 1, figsize=(14, 14))
ax1 = plt.subplot(1, 1, 1, projection='polar')
ax1.set_ylim(0, 5)
ax1.set_title("Orbital Mechanics", fontsize=12, weight='bold', pad=20)
earth_hist, mars_hist, ship_hist = [], [], []
trail_length = 100
graph_area = st.empty()

# Main simulation loop
for step in range(STEPS):
    mission_timer += current_dt
    
    # Verlet integration for planetary positions
    accel_earth = acceleration(r_earth)
    accel_mars = acceleration(r_mars)
    r_earth_next = r_earth + v_earth * current_dt + 0.5 * accel_earth * current_dt**2
    r_mars_next = r_mars + v_mars * current_dt + 0.5 * accel_mars * current_dt**2
    v_earth_next = v_earth + 0.5 * (accel_earth + acceleration(r_earth_next)) * current_dt
    v_mars_next = v_mars + 0.5 * (accel_mars + acceleration(r_mars_next)) * current_dt
    r_earth, v_earth = r_earth_next, v_earth_next
    r_mars, v_mars = r_mars_next, v_mars_next

    theta_e = np.arctan2(r_earth[1], r_earth[0]) % (2 * np.pi)
    theta_m = np.arctan2(r_mars[1], r_mars[0]) % (2 * np.pi)
    earth_hist.append([np.linalg.norm(r_earth), theta_e])
    mars_hist.append([np.linalg.norm(r_mars), theta_m])
    if len(earth_hist) > trail_length: earth_hist.pop(0)
    if len(mars_hist) > trail_length: mars_hist.pop(0)
    
    current_velocity = np.linalg.norm(v_ship)
    max_velocity = max(max_velocity, current_velocity)
    orbital_energy = calculate_orbital_energy(r_ship, v_ship)
    
    if state == "on_earth":
        r_ship, v_ship = r_earth.copy(), v_earth.copy()
        approach_phase = "none"
        current_dt = DEFAULT_DT
        launch_attempt_timer += current_dt
        current_target = "none"
        window_open, phase_error, efficiency = optimal_launch_window(r_earth, r_mars, v_earth, v_mars, "mars")
        launch_window_efficiency = efficiency * 100
        if abs(phase_error) < STRICT_LAUNCH_WINDOW and phase_error <= 0 and launch_attempt_timer > launch_wait_time:
            v_ship = hohmann_velocity(r_e, r_m, r_earth, v_earth)
            state = "transfer_to_mars"
            ship_hist = []
            min_sun_distance = float('inf')
            total_launches += 1
            last_launch_time = mission_timer
            launch_attempt_timer = 0
            transfer_start_time = mission_timer
            mission_start_times.append(mission_timer)
    
    elif state == "transfer_to_mars":
        current_transfer_time = mission_timer - transfer_start_time
        sun_dist = np.linalg.norm(r_ship)
        if sun_dist < SUN_DANGER_RADIUS:
            sun_safety_violations += 1
            state = "emergency_abort"
            break
        accel_ship = acceleration(r_ship)
        r_ship_next = r_ship + v_ship * current_dt + 0.5 * accel_ship * current_dt**2
        v_ship_next = v_ship + 0.5 * (accel_ship + acceleration(r_ship_next)) * current_dt
        r_ship, v_ship = r_ship_next, v_ship_next
        dist_to_mars = np.linalg.norm(r_ship - r_mars)
        update_proximity_tracking(r_ship, r_mars, "Mars")
        if dist_to_mars < APPROACH_THRESHOLD and approach_phase == "none":
            approach_phase = "far_approach"
            current_dt = APPROACH_DT
        elif dist_to_mars < 0.20 and approach_phase == "far_approach":
            approach_phase = "close_approach"
        elif dist_to_mars < 0.10 and approach_phase == "close_approach":
            approach_phase = "final_capture"
        if approach_phase != "none":
            velocity_correction = adaptive_approach_burn(r_ship, v_ship, r_mars, v_mars, r_m, approach_phase)
            v_ship += velocity_correction * current_dt
        min_sun_distance = min(min_sun_distance, sun_dist)
        landing_speed = calculate_relative_velocity(v_ship, v_mars)
        theta_s = np.arctan2(r_ship[1], r_ship[0]) % (2 * np.pi)
        ship_hist.append([np.linalg.norm(r_ship), theta_s])
        if dist_to_mars < ARRIVAL_THRESHOLD and landing_speed < 0.18:
            mars_tangent = np.array([-r_mars[1], r_mars[0]]) / np.linalg.norm(r_mars)
            v_ship = np.sqrt(G * M / r_m) * mars_tangent
            state = "on_mars"
            mars_visits += 1
            ship_hist = []
            approach_phase = "none"
            current_dt = DEFAULT_DT
    
    elif state == "on_mars":
        r_ship, v_ship = r_mars.copy(), v_mars.copy()
        approach_phase = "none"
        current_dt = DEFAULT_DT
        launch_attempt_timer += current_dt
        current_target = "none"
        window_open, phase_error, efficiency = optimal_launch_window(r_earth, r_mars, v_earth, v_mars, "earth")
        launch_window_efficiency = efficiency * 100
        if window_open and launch_attempt_timer > launch_wait_time:
            v_ship = hohmann_velocity(r_m, r_e, r_mars, v_mars)
            state = "transfer_to_earth"
            ship_hist = []
            min_sun_distance = float('inf')
            last_launch_time = mission_timer
            launch_attempt_timer = 0
            transfer_start_time = mission_timer
    
    elif state == "transfer_to_earth":
        current_transfer_time = mission_timer - transfer_start_time
        sun_dist = np.linalg.norm(r_ship)
        if sun_dist < SUN_DANGER_RADIUS:
            sun_safety_violations += 1
            state = "emergency_abort"
            break
        accel_ship = acceleration(r_ship)
        r_ship_next = r_ship + v_ship * current_dt + 0.5 * accel_ship * current_dt**2
        v_ship_next = v_ship + 0.5 * (accel_ship + acceleration(r_ship_next)) * current_dt
        r_ship, v_ship = r_ship_next, v_ship_next
        dist_to_earth = np.linalg.norm(r_ship - r_earth)
        update_proximity_tracking(r_ship, r_earth, "Earth")
        if dist_to_earth < APPROACH_THRESHOLD and approach_phase == "none":
            approach_phase = "far_approach"
            current_dt = APPROACH_DT
        elif dist_to_earth < 0.30 and approach_phase == "far_approach":
            approach_phase = "close_approach"
        elif dist_to_earth < 0.12 and approach_phase == "close_approach":
            approach_phase = "final_capture"
        if approach_phase != "none":
            velocity_correction = adaptive_approach_burn(r_ship, v_ship, r_earth, v_earth, r_e, approach_phase)
            v_ship += velocity_correction * current_dt
        min_sun_distance = min(min_sun_distance, sun_dist)
        landing_speed = calculate_relative_velocity(v_ship, v_earth)
        theta_s = np.arctan2(r_ship[1], r_ship[0]) % (2 * np.pi)
        ship_hist.append([np.linalg.norm(r_ship), theta_s])
        if dist_to_earth < ARRIVAL_THRESHOLD and landing_speed < 0.25:
            earth_tangent = np.array([-r_earth[1], r_earth[0]]) / np.linalg.norm(r_earth)
            v_ship = np.sqrt(G * M / r_e) * earth_tangent
            state = "on_earth"
            successful_missions += 1
            mission_durations.append(mission_timer - mission_start_times[-1])
            ship_hist = []
            approach_phase = "none"
            current_dt = DEFAULT_DT
    
    # Sidebar updates
    if step % 10 == 0:
        status_icon = {"on_earth": "🌍", "on_mars": "🔴", "transfer_to_mars": "🚀→🔴", "transfer_to_earth": "🚀→🌍"}.get(state, "🚀")
        sidebar_mission_status.markdown(f"""
        **{status_icon} MISSION STATUS**
        - **Time:** {mission_timer * CONVERSION_FACTOR:.1f} months
        - **Phase:** {state.upper().replace('_', ' ')}
        - **Velocity:** {current_velocity:.4f} AU/t
        - **Max Velocity:** {max_velocity:.4f} AU/t
        """)
        
        if state in ["on_earth", "on_mars"]:
            destination = "Mars" if state == "on_earth" else "Earth"
            window_open, phase_error, efficiency = optimal_launch_window(r_earth, r_mars, v_earth, v_mars, destination.lower())
            window_status = "🟢 OPEN" if window_open else "🔴 CLOSED"
            sidebar_launch_window.markdown(f"""
            **🎯 LAUNCH WINDOW - {destination}**
            - **Status:** {window_status}
            - **Efficiency:** {efficiency*100:.1f}%
            - **Phase Error:** {np.degrees(phase_error):.2f}°
            - **Wait Time:** {launch_attempt_timer * CONVERSION_FACTOR:.1f} months
            """)
        else:
            sidebar_launch_window.markdown(f"""
            **🚀 TRANSFER TRAJECTORY**
            - **Transfer Time:** {current_transfer_time * CONVERSION_FACTOR:.1f} months
            - **Target:** {current_target}
            - **Approach Phase:** {approach_phase.upper().replace('_', ' ')}
            - **Landing Speed:** {landing_speed:.4f} AU/t
            """)
        
        sun_dist_display = np.linalg.norm(r_ship)
        safety_margin = ((sun_dist_display - SUN_DANGER_RADIUS) / SUN_DANGER_RADIUS * 100) if sun_dist_display > SUN_DANGER_RADIUS else -100
        safety_status = "🟢 OPTIMAL" if safety_margin > 20 else "🟡 SAFE" if safety_margin > 0 else "🔴 CRITICAL"
        sidebar_trajectory.markdown(f"""
        **🛰️ NAVIGATION DATA**
        - **Sun Distance:** {sun_dist_display:.4f} AU
        - **Safety Margin:** {safety_margin:.1f}% ({safety_status})
        - **Orbital Energy:** {orbital_energy:.4f} units
        - **Trajectory Accuracy:** {trajectory_accuracy:.1f}%
        """)
        
        success_rate = (successful_missions / total_launches * 100) if total_launches > 0 else 0
        avg_mission_duration = np.mean(mission_durations) * CONVERSION_FACTOR if mission_durations else 0
        sidebar_performance.markdown(f"""
        **📊 PERFORMANCE METRICS**
        - **Success Rate:** {success_rate:.1f}%
        - **Completed:** {successful_missions}/4 missions
        - **Mars Visits:** {mars_visits}
        - **Avg Duration:** {avg_mission_duration:.1f} months
        """)
        
        if current_target != "none" and approach_phase != "none":
            current_distance_to_target = np.linalg.norm(r_ship - (r_mars if current_target == "Mars" else r_earth))
            approach_progress = max(0, 100 * (1 - current_distance_to_target / APPROACH_THRESHOLD))
            proximity_status = "🟢 EXCELLENT" if min_proximity_to_target < 0.03 else "🟡 GOOD" if min_proximity_to_target < 0.05 else "🟠 MODERATE"
            sidebar_proximity.markdown(f"""
            **🎯 PROXIMITY ANALYSIS**
            - **Target:** {current_target}
            - **Distance:** {current_distance_to_target:.4f} AU
            - **Min Achieved:** {min_proximity_to_target:.4f} AU
            - **Approach:** {approach_progress:.1f}% ({proximity_status})
            """)
        else:
            sidebar_proximity.markdown(f"""
            **🎯 PROXIMITY MONITORING**
            - **Mode:** CRUISE/STANDBY
            - **Target:** {current_target if current_target != "none" else "N/A"}
            - **Status:** MONITORING
            - **Phase:** NAVIGATION
            """)
        
        fuel_efficiency = max(0, 100 - (fuel_used / max(1, total_launches) * 8))
        avg_delta_v = total_delta_v / max(1, total_launches)
        sidebar_safety.markdown(f"""
        **⚠️ SAFETY & EFFICIENCY**
        - **Fuel Used:** {fuel_used:.3f} units
        - **Fuel Efficiency:** {fuel_efficiency:.1f}%
        - **Total ΔV:** {total_delta_v:.3f} AU/t
        - **Safety Violations:** {sun_safety_violations}
        """)
        
        completion_progress = (successful_missions / 4 * 100)
        time_efficiency = (successful_missions / max(1, (mission_timer * CONVERSION_FACTOR) / 8.5) * 100) if mission_timer > 0 else 0
        sidebar_completion.markdown(f"""
        **🏆 MISSION COMPLETION**
        - **Progress:** {completion_progress:.1f}% (4 missions)
        - **Time Efficiency:** {time_efficiency:.1f}%
        - **Launch Efficiency:** {launch_window_efficiency:.1f}%
        - **Status:** {'🎉 COMPLETE' if completion_progress >= 100 else '⏳ IN PROGRESS'}
        """)
        
        ax1.clear()
        ax1.set_ylim(0, 5)
        ax1.set_title("Orbital Mechanics", fontsize=16, weight='bold', pad=20)
        ax1.scatter(0, 0, color='gold', s=500, label="Sun", zorder=10)
        theta_danger = np.linspace(0, 2 * np.pi, 100)
        danger_radius_plot = np.full_like(theta_danger, SUN_DANGER_RADIUS)
        ax1.fill_between(theta_danger, 0, danger_radius_plot, color='red', alpha=0.12, zorder=1)
        ax1.plot(theta_danger, danger_radius_plot, color='red', linestyle='--', linewidth=1, zorder=2)
        if len(earth_hist) > 8:
            recent_earth = np.array(earth_hist)
            ax1.plot(recent_earth[:, 1], recent_earth[:, 0], 'b-', alpha=0.6, linewidth=2.8, label="Earth Orbit")
        ax1.plot([theta_e], [np.linalg.norm(r_earth)], 'bo', markersize=17, label="Earth")
        if len(mars_hist) > 8:
            recent_mars = np.array(mars_hist)
            ax1.plot(recent_mars[:, 1], recent_mars[:, 0], 'r-', alpha=0.6, linewidth=2.8, label="Mars Orbit")
        ax1.plot([theta_m], [np.linalg.norm(r_mars)], 'ro', markersize=15, label="Mars")
        if len(ship_hist) > 0:
            ship_array = np.array(ship_hist)
            ax1.plot(ship_array[:, 1], ship_array[:, 0], 'g-', linewidth=5.5, alpha=0.95, label="Spacecraft")
            ax1.plot([ship_array[-1, 1]], [ship_array[-1, 0]], 'gs', markersize=15)
        else:
            current_ship_angle = theta_e if state in ["on_earth", "transfer_to_earth"] else theta_m
            current_ship_radius = np.linalg.norm(r_earth if state in ["on_earth", "transfer_to_earth"] else r_mars)
            ax1.plot([current_ship_angle], [current_ship_radius], 'gs', markersize=15, label="Spacecraft")
        ax1.legend(loc='center left', bbox_to_anchor=(1.08, 0.5))
        graph_area.pyplot(fig, clear_figure=False)
        if successful_missions >= 4: break

# Final mission summary
if state == "emergency_abort":
    st.error("💀 MISSION CRITICAL FAILURE: Spacecraft lost to solar radiation!")
    st.warning(f"⚠️ Critical approach distance to Sun: **{min_sun_distance:.4f} AU**")
else:
    if successful_missions >= 4:
        st.success(f"🎉 MISSION EXCELLENCE! **{successful_missions}** efficient round-trip missions completed!")
    elif successful_missions > 0:
        st.success(f"🎉 MISSION SUCCESS! **{successful_missions}** efficient round-trip missions completed!")
    else:
        st.warning("🟡 Mission ended before 4 successful round-trips or no missions completed.")
        st.info("⏳ Simulation complete or ended early.")
    
    success_rate_final = (successful_missions / total_launches * 100) if total_launches > 0 else 0
    fuel_efficiency_final = max(0, 100 - (fuel_used / max(1, total_launches) * 8))
    avg_mission_time = np.mean(mission_durations) * CONVERSION_FACTOR if mission_durations else 0
    
    st.info(f"📊 **Final Performance Summary:**")
    st.info(f"🏆 Success Rate: **{success_rate_final:.1f}%** | ⛽ Fuel Efficiency: **{fuel_efficiency_final:.1f}%** | ⏱️ Avg Mission Time: **{avg_mission_time:.1f}** months")
    
    if min_proximity_to_target != float('inf') and current_target != "none":
        st.info(f"🎯 Best proximity to {current_target}: **{min_proximity_to_target:.4f} AU**")
    elif min_sun_distance != float('inf'):
        st.info(f"☀️ Closest Sun approach: **{min_sun_distance:.4f} AU** | 🚀 Max velocity: **{max_velocity:.4f} AU/t**")