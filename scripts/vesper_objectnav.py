#!/usr/bin/env python3
"""
VESPER ObjectNav Demo - First-person navigation in photorealistic indoor scenes.
Demonstrates "Go to object/room" style navigation like the Habitat AI demo.
"""

import os
import sys

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "habitat-lab-official"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "habitat-lab-official", "habitat-lab"))

import numpy as np
import magnum as mn
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from habitat_sim.errors import GreedyFollowerError
import pygame
import random
from typing import Optional, Dict, List, Tuple
import json

# High-quality rendering config
RESOLUTION = (1280, 720)  # HD resolution
SENSOR_HEIGHT = 1.5  # Human eye height
MOVE_SPEED = 0.15  # m per step
TURN_SPEED = 5.0  # degrees per step

# Third-person camera settings
THIRD_PERSON_DISTANCE = 3.0
THIRD_PERSON_HEIGHT = 2.0


class ObjectNavDemo:
    """First-person navigation demo with object/room goals."""
    
    def __init__(self):
        self.sim: Optional[habitat_sim.Simulator] = None
        self.agent = None
        self.path_follower = None  # GreedyGeodesicFollower
        self.current_goal = None
        self.objects_in_scene: Dict[str, List[mn.Vector3]] = {}
        self.humanoid = None  # Articulated object for humanoid
        self.use_third_person = False  # Start in first-person view
        
        # Navigation target types (rooms/objects)
        self.target_types = [
            "toilet", "bed", "couch", "chair", "dining table",
            "refrigerator", "tv", "sink", "bathtub", "kitchen"
        ]
        
    def find_scene(self) -> Tuple[str, Optional[str]]:
        """Find a scene file to load. Returns (scene_path, config_path)."""
        data_path = os.path.join(PROJECT_ROOT, "data")
        
        # Try HSSD-hab scenes first (high quality with semantics support)
        hssd_path = os.path.join(data_path, "scene_datasets", "hssd-hab", "scenes")
        if os.path.exists(hssd_path):
            scenes = [f for f in os.listdir(hssd_path) if f.endswith(".scene_instance.json")]
            if scenes:
                scene = random.choice(scenes[:10])
                config_path = os.path.join(os.path.dirname(hssd_path), "hssd-hab.scene_dataset_config.json")
                return os.path.join(hssd_path, scene), config_path
        
        # Try HM3D scenes with semantic annotations
        hm3d_path = os.path.join(data_path, "scene_datasets", "hm3d", "example")
        if os.path.exists(hm3d_path):
            config_path = os.path.join(hm3d_path, "hm3d_annotated_example_basis.scene_dataset_config.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(hm3d_path, "hm3d_annotated_basis.scene_dataset_config.json")
            
            for subdir in os.listdir(hm3d_path):
                scene_dir = os.path.join(hm3d_path, subdir)
                if os.path.isdir(scene_dir):
                    for f in os.listdir(scene_dir):
                        if f.endswith(".basis.glb"):
                            scene_file = os.path.join(scene_dir, f)
                            return scene_file, config_path if os.path.exists(config_path) else None
        
        # Try ReplicaCAD
        replica_path = os.path.join(data_path, "replica_cad", "configs", "scenes")
        if os.path.exists(replica_path):
            scenes = [f for f in os.listdir(replica_path) if f.endswith(".scene_instance.json")]
            if scenes:
                config = os.path.join(data_path, "replica_cad", "replicaCAD.scene_dataset_config.json")
                return os.path.join(replica_path, random.choice(scenes)), config
                
        raise FileNotFoundError("No scene files found! Please download a dataset.")
    
    def create_simulator(self, scene_path: str, config_path: Optional[str] = None) -> habitat_sim.Simulator:
        """Create simulator with high-quality rendering settings."""
        
        # Backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = True
        backend_cfg.physics_config_file = os.path.join(
            PROJECT_ROOT, "data", "default.physics_config.json"
        )
        
        # Use provided config or search for one
        if config_path and os.path.exists(config_path):
            backend_cfg.scene_dataset_config_file = config_path
            print(f"Using scene dataset config: {os.path.basename(config_path)}")
        
        # Agent configuration with high-res sensors
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = SENSOR_HEIGHT
        agent_cfg.radius = 0.1
        
        # RGB sensor (first-person view)
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "rgb"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [RESOLUTION[1], RESOLUTION[0]]
        rgb_sensor.position = [0.0, SENSOR_HEIGHT, 0.0]
        rgb_sensor.hfov = 90  # Wide FOV for better spatial awareness
        
        # Depth sensor for obstacle awareness
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [RESOLUTION[1], RESOLUTION[0]]
        depth_sensor.position = [0.0, SENSOR_HEIGHT, 0.0]
        depth_sensor.hfov = 90
        
        # Semantic sensor for object detection
        semantic_sensor = habitat_sim.CameraSensorSpec()
        semantic_sensor.uuid = "semantic"
        semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor.resolution = [RESOLUTION[1] // 4, RESOLUTION[0] // 4]  # Lower res
        semantic_sensor.position = [0.0, SENSOR_HEIGHT, 0.0]
        semantic_sensor.hfov = 90
        
        # Third-person camera (looking at agent from behind and above)
        third_person_sensor = habitat_sim.CameraSensorSpec()
        third_person_sensor.uuid = "third_rgb"
        third_person_sensor.sensor_type = habitat_sim.SensorType.COLOR
        third_person_sensor.resolution = [RESOLUTION[1], RESOLUTION[0]]
        third_person_sensor.position = [0.0, THIRD_PERSON_HEIGHT, THIRD_PERSON_DISTANCE]
        third_person_sensor.orientation = [-0.3, 0.0, 0.0]  # Look down slightly
        third_person_sensor.hfov = 90
        
        agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor, semantic_sensor, third_person_sensor]
        
        # Action space
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=MOVE_SPEED)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=MOVE_SPEED)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_SPEED)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_SPEED)
            ),
            "look_up": habitat_sim.agent.ActionSpec(
                "look_up", habitat_sim.agent.ActuationSpec(amount=TURN_SPEED)
            ),
            "look_down": habitat_sim.agent.ActionSpec(
                "look_down", habitat_sim.agent.ActuationSpec(amount=TURN_SPEED)
            ),
        }
        
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        sim = habitat_sim.Simulator(cfg)
        
        # Initialize navmesh for pathfinding
        if not sim.pathfinder.is_loaded:
            print("Computing navmesh...")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = SENSOR_HEIGHT
            navmesh_settings.agent_radius = 0.1
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        
        return sim
    
    def init_path_follower(self):
        """Initialize the GreedyGeodesicFollower for proper pathfinding navigation."""
        # This is what Habitat actually uses for ShortestPathFollower
        self.path_follower = self.sim.make_greedy_follower(
            agent_id=0,
            goal_radius=0.5,  # Stop when within 0.5m of goal
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        print("Initialized GreedyGeodesicFollower for navigation")
    
    def get_random_navigable_point(self) -> mn.Vector3:
        """Get a random navigable point in the scene."""
        return self.sim.pathfinder.get_random_navigable_point()
    
    def load_humanoid(self):
        """Load a humanoid model at the agent's position."""
        # Disabled for now - URDF loading creates T-pose without animation
        # Would need proper HumanoidRearrangeController for animated humanoid
        print("Humanoid disabled - using first-person navigation")
        return None
    
    def update_humanoid_position(self):
        """Update humanoid position to match agent."""
        if self.humanoid is not None:
            agent_state = self.agent.get_state()
            self.humanoid.translation = agent_state.position
            # Note: Rotation update requires magnum Quaternion, skipping for now
    
    def find_path_to_goal(self, goal_pos: mn.Vector3) -> Optional[habitat_sim.ShortestPath]:
        """Find shortest path to goal."""
        agent_state = self.agent.get_state()
        start_pos = agent_state.position
        
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos
        
        if self.sim.pathfinder.find_path(path):
            return path
        return None
    
    def get_action_to_goal(self, goal_pos: mn.Vector3) -> Optional[str]:
        """Get the next action to move toward goal using GreedyGeodesicFollower.
        
        This is the same approach Habitat uses in ShortestPathFollower.
        It uses navmesh pathfinding for proper obstacle avoidance.
        """
        if self.path_follower is None:
            print("[NAV] Path follower not initialized!")
            return None
        
        # Check distance to goal first
        agent_state = self.agent.get_state()
        current_pos = np.array(agent_state.position)
        goal = np.array(goal_pos)
        distance = np.linalg.norm(goal - current_pos)
        
        print(f"[NAV] Distance to goal: {distance:.2f}m")
        
        if distance < 0.5:
            print("[NAV] Goal reached!")
            return None
        
        try:
            # Use GreedyGeodesicFollower to get next action
            # This handles all the navmesh pathfinding automatically
            next_action = self.path_follower.next_action_along(goal_pos)
            
            if next_action == "stop" or next_action is None:
                print("[NAV] Follower says stop/None - goal reached or unreachable")
                return None
            
            print(f"[NAV] GreedyFollower action: {next_action}")
            return next_action
            
        except GreedyFollowerError as e:
            print(f"[NAV] GreedyFollowerError: {e}")
            return None
        except Exception as e:
            print(f"[NAV] Error getting action: {e}")
            return None
    
    def get_semantic_objects(self) -> Dict[str, List[Tuple[int, str]]]:
        """Get list of objects in scene with their semantic IDs."""
        objects = {}
        scene = self.sim.semantic_scene
        
        if scene is None:
            return objects
        
        for obj in scene.objects:
            if obj is None:
                continue
            category = obj.category
            if category is None:
                continue
            name = category.name().lower()
            if name not in objects:
                objects[name] = []
            objects[name].append((obj.semantic_id, name))
        
        return objects

    def get_object_positions(self) -> Dict[str, List[mn.Vector3]]:
        """Get positions of semantic objects in the scene."""
        positions = {}
        scene = self.sim.semantic_scene
        
        if scene is None:
            print("No semantic scene available")
            return positions
        
        for obj in scene.objects:
            if obj is None:
                continue
            category = obj.category
            if category is None:
                continue
            name = category.name().lower()
            
            # Get object center (AABB center)
            aabb = obj.aabb
            if aabb is not None:
                center = aabb.center
                # Snap to navmesh if possible
                nav_point = self.sim.pathfinder.snap_point(center)
                if not np.isnan(nav_point[0]):
                    if name not in positions:
                        positions[name] = []
                    positions[name].append(mn.Vector3(nav_point))
        
        return positions
    
    def get_room_positions(self) -> Dict[str, List[mn.Vector3]]:
        """Get navigable positions in different rooms."""
        rooms = {}
        scene = self.sim.semantic_scene
        
        if scene is None:
            return rooms
        
        for region in scene.regions:
            if region is None:
                continue
            category = region.category
            if category is None:
                continue
            name = category.name().lower()
            
            # Get region center
            aabb = region.aabb
            if aabb is not None:
                center = aabb.center
                nav_point = self.sim.pathfinder.snap_point(center)
                if not np.isnan(nav_point[0]):
                    if name not in rooms:
                        rooms[name] = []
                    rooms[name].append(mn.Vector3(nav_point))
        
        return rooms
    
    def load_hssd_semantics(self, scene_path: str) -> Tuple[Dict[str, List[mn.Vector3]], Dict[str, List[mn.Vector3]]]:
        """Load semantics from HSSD semantic config files."""
        rooms = {}
        objects = {}
        
        # Get scene ID from path
        scene_name = os.path.basename(scene_path)
        scene_id = scene_name.split('.')[0]  # e.g., "102343992"
        
        # Look for semantic config
        data_path = os.path.join(PROJECT_ROOT, "data", "scene_datasets", "hssd-hab")
        semantic_config_path = os.path.join(data_path, "semantics", "scenes", f"{scene_id}.semantic_config.json")
        
        if not os.path.exists(semantic_config_path):
            print(f"No semantic config found at {semantic_config_path}")
            return rooms, objects
        
        try:
            with open(semantic_config_path, 'r') as f:
                semantic_data = json.load(f)
        except Exception as e:
            print(f"Error loading semantic config: {e}")
            return rooms, objects
        
        # Extract room/region annotations
        if "region_annotations" in semantic_data:
            for region in semantic_data["region_annotations"]:
                name = region.get("name", "unknown").lower()
                
                # Get center of region from bounds
                min_bounds = region.get("min_bounds")
                max_bounds = region.get("max_bounds")
                
                if min_bounds and max_bounds:
                    center = [
                        (min_bounds[0] + max_bounds[0]) / 2,
                        (min_bounds[1] + max_bounds[1]) / 2,
                        (min_bounds[2] + max_bounds[2]) / 2
                    ]
                    
                    # Snap to navmesh
                    nav_point = self.sim.pathfinder.snap_point(center)
                    if not np.isnan(nav_point[0]):
                        if name not in rooms:
                            rooms[name] = []
                        rooms[name].append(mn.Vector3(nav_point))
                        print(f"  Found room: {name}")
        
        print(f"Loaded {len(rooms)} room types from semantic config")
        return rooms, objects
    
    def get_scene_objects_from_rigid_objects(self) -> Dict[str, List[mn.Vector3]]:
        """Get objects from the simulator's rigid object manager."""
        objects = {}
        
        rom = self.sim.get_rigid_object_manager()
        
        for obj_id in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_id)
            if obj is not None:
                # Get object name from handle
                handle_name = obj_id.split('/')[-1].split('.')[0].lower()
                
                # Only include recognizable furniture/appliance keywords
                recognized_name = None
                for keyword in ['chair', 'table', 'sofa', 'couch', 'bed', 'desk', 'cabinet', 
                               'refrigerator', 'fridge', 'tv', 'television', 'lamp', 
                               'toilet', 'sink', 'bathtub', 'shower', 'oven', 'microwave']:
                    if keyword in handle_name:
                        recognized_name = keyword
                        break
                
                # Skip unrecognized objects (like random asset IDs)
                if recognized_name is None:
                    continue
                
                pos = obj.translation
                nav_point = self.sim.pathfinder.snap_point(pos)
                
                if not np.isnan(nav_point[0]):
                    if recognized_name not in objects:
                        objects[recognized_name] = []
                    objects[recognized_name].append(mn.Vector3(nav_point))
        
        return objects


class GameUI:
    """Pygame UI for ObjectNav demo."""
    
    def __init__(self, nav_demo: ObjectNavDemo):
        self.nav_demo = nav_demo
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = None
        self.small_font = None
        
        # UI State
        self.show_help = True
        self.show_map = False
        self.auto_navigate = False
        self.goal_name = None
        self.goal_pos = None
        self.status_message = "Press T to set room goal, N to auto-navigate"
        self.available_targets = []
        self.third_person = False  # Start in first-person view
        
    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption("VESPER ObjectNav - Navigate to Rooms")
        self.screen = pygame.display.set_mode(RESOLUTION)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def render_frame(self, observations: dict):
        """Render the current frame."""
        # Get RGB observation (first or third person)
        if self.third_person and "third_rgb" in observations:
            rgb = observations["third_rgb"]
        else:
            rgb = observations["rgb"]
        
        # Convert to pygame surface
        # RGB is (H, W, 4) with RGBA
        rgb = rgb[:, :, :3]  # Remove alpha
        rgb = np.ascontiguousarray(rgb)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        
        # Blit to screen
        self.screen.blit(surface, (0, 0))
        
        # Draw UI overlay
        self._draw_overlay()
        
        pygame.display.flip()
        self.clock.tick(60)
        
    def _draw_overlay(self):
        """Draw UI overlay."""
        # Semi-transparent panel at top
        panel = pygame.Surface((RESOLUTION[0], 60), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (0, 0))
        
        # Title
        title = self.font.render("VESPER ObjectNav", True, (0, 255, 255))
        self.screen.blit(title, (20, 10))
        
        # Status message
        status = self.small_font.render(self.status_message, True, (255, 255, 255))
        self.screen.blit(status, (20, 35))
        
        # Goal indicator
        if self.goal_name:
            goal_text = self.font.render(f"Goal: {self.goal_name}", True, (0, 255, 0))
            self.screen.blit(goal_text, (RESOLUTION[0] - 300, 10))
        
        # Auto-nav indicator
        if self.auto_navigate:
            auto_text = self.small_font.render("[AUTO-NAV ON]", True, (255, 200, 0))
            self.screen.blit(auto_text, (RESOLUTION[0] - 300, 35))
        
        # Help panel
        if self.show_help:
            self._draw_help()
    
    def _draw_help(self):
        """Draw help panel."""
        help_panel = pygame.Surface((280, 220), pygame.SRCALPHA)
        help_panel.fill((0, 0, 0, 180))
        
        help_lines = [
            "Controls:",
            "W/↑ - Move Forward",
            "S/↓ - Move Backward", 
            "A/← - Turn Left",
            "D/→ - Turn Right",
            "G - Set random goal",
            "T - Go to object/room",
            "N - Auto-navigate to goal",
            "V - Toggle 1st/3rd person",
            "H - Toggle help",
            "ESC - Quit"
        ]
        
        y = 10
        for line in help_lines:
            text = self.small_font.render(line, True, (255, 255, 255))
            help_panel.blit(text, (10, y))
            y += 22
        
        self.screen.blit(help_panel, (RESOLUTION[0] - 290, 70))
    
    def handle_events(self) -> Optional[str]:
        """Handle pygame events. Returns action or None."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_g:
                    return "set_goal"
                elif event.key == pygame.K_t:
                    return "set_object_goal"
                elif event.key == pygame.K_n:
                    self.auto_navigate = not self.auto_navigate
                    if self.auto_navigate and self.goal_pos is None:
                        self.status_message = "Set a goal first with T"
                        self.auto_navigate = False
                elif event.key == pygame.K_v:
                    self.third_person = not self.third_person
                    view_name = "Third-person" if self.third_person else "First-person"
                    self.status_message = f"{view_name} view"
        
        # Continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            return "move_forward"
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            return "move_backward"
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            return "turn_left"
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            return "turn_right"
        elif keys[pygame.K_q]:
            return "look_up"
        elif keys[pygame.K_e]:
            return "look_down"
        
        return None
    
    def set_goal(self, name: str, pos: mn.Vector3, distance: float):
        """Set current navigation goal."""
        self.goal_name = name
        self.goal_pos = pos
        self.status_message = f"Navigate to {name} ({distance:.1f}m away)"
        
    def clear_goal(self):
        """Clear current goal."""
        self.goal_name = None
        self.goal_pos = None
        self.auto_navigate = False
        self.status_message = "Goal reached! Press G for new goal"


def main():
    """Main entry point."""
    print("=" * 60)
    print("VESPER ObjectNav Demo")
    print("First-person navigation in photorealistic indoor scenes")
    print("=" * 60)
    
    # Create demo
    demo = ObjectNavDemo()
    
    # Find and load scene
    print("\nFinding scene...")
    scene_path, config_path = demo.find_scene()
    print(f"Loading scene: {os.path.basename(scene_path)}")
    
    # Create simulator
    demo.sim = demo.create_simulator(scene_path, config_path)
    demo.agent = demo.sim.get_agent(0)
    
    # Initialize the GreedyGeodesicFollower for proper navigation
    demo.init_path_follower()
    
    # Place agent at navigable point
    start_pos = demo.get_random_navigable_point()
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_pos
    demo.agent.set_state(agent_state)
    
    print(f"Agent starting at: {start_pos}")
    
    # Get semantic info
    semantic_objects = demo.get_semantic_objects()
    object_positions = demo.get_object_positions()
    room_positions = demo.get_room_positions()
    
    # Try to load HSSD semantics
    if "hssd" in scene_path.lower():
        hssd_rooms, hssd_objects = demo.load_hssd_semantics(scene_path)
        room_positions.update(hssd_rooms)
        object_positions.update(hssd_objects)
    
    # Get objects from rigid object manager
    rigid_objects = demo.get_scene_objects_from_rigid_objects()
    object_positions.update(rigid_objects)
    
    # Combine objects and rooms
    all_targets = {**object_positions, **room_positions}
    
    print(f"Found {len(semantic_objects)} object categories in scene")
    for cat in list(semantic_objects.keys())[:10]:
        print(f"  - {cat}: {len(semantic_objects[cat])} instances")
    
    if room_positions:
        print(f"Found {len(room_positions)} room types:")
        for room in room_positions:
            print(f"  - {room}")
    
    # Create UI
    ui = GameUI(demo)
    ui.init_pygame()
    ui.available_targets = list(all_targets.keys()) if all_targets else []
    
    # Try to load humanoid for third-person view
    print("\nLoading humanoid...")
    demo.load_humanoid()
    
    print("\nStarting navigation demo...")
    print("Controls: WASD to move, T for room goal, N to auto-navigate, V to toggle view, ESC to quit")
    
    running = True
    while running:
        # Get observations
        obs = demo.sim.get_sensor_observations()
        
        # Render
        ui.render_frame(obs)
        
        # Handle input
        action = ui.handle_events()
        
        if action == "quit":
            running = False
        elif action == "set_goal":
            # Set random navigable point as goal
            goal = demo.get_random_navigable_point()
            agent_state = demo.agent.get_state()
            distance = np.linalg.norm(
                np.array(goal) - np.array(agent_state.position)
            )
            ui.set_goal("Random Location", goal, distance)
            ui.goal_pos = goal
        elif action == "set_object_goal":
            # Prioritize room navigation over objects
            if room_positions:
                target_type = random.choice(list(room_positions.keys()))
                target_pos = random.choice(room_positions[target_type])
                agent_state = demo.agent.get_state()
                distance = np.linalg.norm(
                    np.array(target_pos) - np.array(agent_state.position)
                )
                ui.set_goal(f"Go to {target_type}", target_pos, distance)
                ui.goal_pos = target_pos
                print(f"[ObjectNav] Target room: {target_type} at {target_pos}, distance: {distance:.1f}m")
            elif all_targets:
                target_type = random.choice(list(all_targets.keys()))
                target_pos = random.choice(all_targets[target_type])
                agent_state = demo.agent.get_state()
                distance = np.linalg.norm(
                    np.array(target_pos) - np.array(agent_state.position)
                )
                ui.set_goal(f"Go to {target_type}", target_pos, distance)
                ui.goal_pos = target_pos
                print(f"[ObjectNav] Target: {target_type} at {target_pos}")
            else:
                ui.status_message = "No rooms or objects available"
                print("[ObjectNav] No targets available!")
        elif action in ["move_forward", "move_backward", "turn_left", "turn_right", 
                        "look_up", "look_down"]:
            demo.agent.act(action)
            demo.update_humanoid_position()  # Keep humanoid in sync
        
        # Auto-navigation runs independently (not in elif)
        if ui.auto_navigate and ui.goal_pos is not None and action is None:
            auto_action = demo.get_action_to_goal(ui.goal_pos)
            if auto_action:
                demo.agent.act(auto_action)
                demo.update_humanoid_position()
                # Update distance display
                agent_state = demo.agent.get_state()
                distance = np.linalg.norm(
                    np.array(ui.goal_pos) - np.array(agent_state.position)
                )
                ui.status_message = f"Navigating... {distance:.1f}m remaining"
            else:
                ui.clear_goal()
                print("[ObjectNav] Goal reached!")
    
    # Cleanup
    demo.sim.close()
    pygame.quit()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
