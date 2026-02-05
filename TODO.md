# VESPER Development TODO

## ✅ RESOLVED ISSUES

### Issue 1: Humanoid Not Visible in Third-Person View
**Status**: ✅ FIXED
**Solution**: Used Habitat-Lab's `KinematicHumanoid` class with `HumanoidRearrangeController` for walking animations
- Loaded female_0.urdf + female_0_motion_data_smplx.pkl from data/versioned_data/habitat_humanoids/
- Fixed third-person camera position: +2.5m Z (behind agent), 1.5m height
- Added 180° rotation offset so humanoid faces away from camera (proper 3rd person view)
- Walking animations working via `calculate_walk_pose()` and `set_joint_transform()`

### Issue 2: Security Camera Feed Wrong Angle
**Status**: ✅ FIXED
**Solution**: Implemented scene-aware placement with proper pan/tilt calculation
- Created `vesper/devices/security_camera.py` with auto-angle calculation
- Created `vesper/devices/scene_device_placer.py` for room corner placement
- Cameras and motion sensors co-located at room corners with 90° coverage

### Issue 3: Third-Person Camera Positioning
**Status**: ✅ FIXED
**Solution**: Fixed coordinate system - camera at +Z (behind agent who faces -Z)
- THIRD_PERSON_DISTANCE = 2.5m (positive Z = behind)
- THIRD_PERSON_HEIGHT = 1.5m (eye level)
- Tilt: -0.15 radians (~8.5° down)

---

## Phase 5: Realistic IoT Sensors & Humanoid Embodiment

### 5.1 Realistic Motion Sensors
- [x] Create `vesper/habitat/sensors/motion_sensor.py`
  - [x] PIR-style detection cone (not just radius)
  - [x] Configurable detection angle (e.g., 110° typical PIR)
  - [x] Configurable detection range (e.g., 5-12 meters)
  - [x] Mounting height and orientation
  - [x] Cooldown period between detections
  - [x] Sensitivity levels (high/medium/low)
  - [ ] Visual representation of detection cone in 3D

### 5.2 Security Cameras
- [x] Create `vesper/devices/security_camera.py`
  - [x] Field of view (FOV) configuration
  - [x] Pan/tilt orientation with auto-calculation
  - [x] Track humanoid position (logic works)
  - [x] Capture RGB frames from camera viewpoint
  - [x] Motion detection within camera view
  - [x] Correct camera angle to see floor/humanoid
  - [ ] Visual FOV cone in 3D overlay

### 5.3 Virtual Humanoid Avatar ✅ COMPLETE
- [x] **FIXED**: Humanoid rendering in Habitat-Sim
  - [x] Used Habitat-Lab's KinematicHumanoid class
  - [x] Integrated HumanoidRearrangeController for motion
  - [x] Model loads and renders correctly
- [x] Eye-level camera for first-person view
- [x] Smooth transition between 1st/3rd person (V key)
- [x] Avatar visible in third-person view
- [x] Walking animations from motion capture data

### 5.4 SmartThings Integration
- [x] Create `vesper/integrations/smartthings.py`
  - [x] SmartThings API client
  - [x] Device state synchronization (virtual → real)
  - [x] Event forwarding (virtual events → SmartThings)
  - [x] Device capability mapping
  - [ ] OAuth2 authentication flow (using PAT instead)
  - [x] Rate limiting and error handling

---

## Phase 6: Autonomous Daily Life Simulation (Ultimate Goal)

### 6.1 Time Synchronization
- [x] Create `vesper/simulation/time_manager.py`
  - [x] Real-time clock synchronization
  - [x] Time acceleration mode (for testing)
  - [x] Day/night cycle tracking
  - [x] Time-based event scheduling

### 6.2 Daily Task System
- [x] Create `vesper/simulation/task_system.py`
  - [x] Task definition schema (name, duration, location, actions)
  - [x] Task categories: morning routine, meals, work, leisure, hygiene, sleep
  - [x] Realistic task durations (cooking: 30-60 min, shower: 10-15 min, etc.)
  - [x] Task prerequisites and dependencies
  - [x] Task interruption handling

### 6.3 LLM Task Generation
- [x] Create `vesper/simulation/task_generator.py`
  - [x] Context-aware task generation (time, location, history)
  - [x] Daily schedule generation
  - [x] Personality profiles for humanoids
  - [x] Random events and variations
  - [x] Workday/weekend schedules

### 6.4 Task History Database
- [x] Create `vesper/simulation/task_database.py`
  - [x] SQLite backend for task history
  - [x] Task completion logging
  - [x] Query interface (by date, type, humanoid)
  - [x] Statistics and analytics
  - [x] Export to JSON

### 6.5 Event Stream Generation
- [x] Create `vesper/simulation/event_stream.py`
  - [x] Event type definitions
  - [x] Event publishing/subscription
  - [x] Simulation coordinator
  - [x] Event logging to file
  - [ ] MQTT event output
  - [ ] Anomaly injection for testing

---

## Implementation Order

### Sprint 1: Realistic Sensors ✅ COMPLETE
1. [x] Motion sensor with detection cone
2. [ ] Visual overlay for sensor FOV
3. [x] Camera sensor with tracking

### Sprint 2: Simulation Core ✅ COMPLETE
4. [x] Time manager with real-time sync
5. [x] Task system with factory patterns
6. [x] Task generator with LLM support
7. [x] Task database with SQLite
8. [x] Event stream coordination

### Sprint 3: Humanoid Avatar ✅ COMPLETE
9. [x] First-person eye-level view
10. [x] Third-person avatar rendering with KinematicHumanoid
11. [x] View toggle (V key) - humanoid visible in 3rd person
12. [x] Walking animations via HumanoidRearrangeController

### Sprint 4: SmartThings Integration ✅ COMPLETE
13. [x] API integration
14. [x] Device bridging
15. [x] Event forwarding

### Sprint 5: Integration & Testing ✅ IN PROGRESS
16. [x] Wire sensors to event stream
17. [x] Wire tasks to humanoid navigation
18. [x] Full autonomous demo working

---

## Next Steps (Priority Order)

### Step 1: Polish and Optimize
- [ ] Add idle animation when standing still
- [ ] Improve walking animation sync with movement speed
- [ ] Add collision-aware third-person camera

### Step 2: Additional Features
- [ ] Multiple humanoids in scene
- [ ] Object interaction animations
- [ ] MQTT event output for external systems

### Step 3: Documentation
- [ ] Update README with demo instructions
- [ ] Document API for external integrations
- [ ] Create video demo

---

## Module Structure

```
vesper/
├── devices/               # IoT device models
│   ├── __init__.py
│   ├── security_camera.py     # Security camera with auto-angles
│   ├── scene_device_placer.py # Room corner placement
│   └── scene_configs.py       # Scene type configurations
├── habitat/
│   ├── sensors/           # Sensor models
│   │   ├── __init__.py
│   │   ├── motion_sensor.py   # PIR motion sensor
│   │   └── camera.py          # Security camera
│   └── ...
├── simulation/            # Simulation systems
│   ├── __init__.py
│   ├── time_manager.py    # Real-time synchronization
│   ├── task_system.py     # Task definitions
│   ├── task_generator.py  # LLM task generation
│   ├── task_database.py   # History storage
│   └── event_stream.py    # Continuous event output
├── integrations/          # External integrations
│   ├── __init__.py
│   └── smartthings.py     # SmartThings bridge
└── ...

scripts/
├── vesper_objectnav_camera_humanoid.py  # Main demo with humanoid + cameras
└── ...
```

---

## Notes

- All modules should be self-contained with clear interfaces
- Use dependency injection for testability
- Main entry point: `scripts/vesper_objectnav_camera_humanoid.py`
- Prefer configuration over hardcoding
- Document all public APIs
