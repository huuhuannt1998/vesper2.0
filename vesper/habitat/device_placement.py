"""
Device placement utilities for positioning IoT devices in scenes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vesper.devices.base import IoTDevice
from vesper.core.environment import Environment

logger = logging.getLogger(__name__)


@dataclass
class PlacementConfig:
    """Configuration for device placement."""
    auto_place: bool = True
    motion_sensor_height: float = 2.5
    contact_sensor_height: float = 1.0
    light_sensor_height: float = 2.0


@dataclass
class PlacementRule:
    """Rule for placing a device type."""
    device_type: str
    room_types: List[str]
    height: float
    position_type: str  # "center", "corner", "door", "wall"


class DevicePlacer:
    """
    Utility for placing IoT devices in 3D scenes.
    
    Example:
        placer = DevicePlacer(scene_manager, environment)
        placer.auto_place_devices()
    """
    
    DEFAULT_RULES = [
        PlacementRule("motion_sensor", ["living_room", "kitchen", "hallway"], 2.5, "center"),
        PlacementRule("contact_sensor", ["bedroom", "bathroom"], 1.0, "door"),
        PlacementRule("smart_door", ["entrance", "hallway"], 1.0, "door"),
        PlacementRule("light_sensor", ["living_room", "bedroom"], 2.0, "wall"),
    ]
    
    def __init__(
        self,
        scene_manager: Any = None,
        environment: Optional[Environment] = None,
        config: Optional[PlacementConfig] = None,
    ):
        self.scene_manager = scene_manager
        self.environment = environment
        self.config = config or PlacementConfig()
        self._placements: List[Dict[str, Any]] = []
    
    @property
    def placements(self) -> List[Dict[str, Any]]:
        return self._placements.copy()
    
    def compute_placements(
        self,
        rules: Optional[List[PlacementRule]] = None,
    ) -> List[Dict[str, Any]]:
        """Compute device placements based on rules."""
        rules = rules or self.DEFAULT_RULES
        self._placements = []
        
        if not self.scene_manager or not self.scene_manager.is_loaded:
            logger.warning("Scene not loaded, using default placements")
            return self._get_default_placements()
        
        for rule in rules:
            rooms = []
            for room_type in rule.room_types:
                rooms.extend(self.scene_manager.get_room_by_type(room_type))
            
            for room in rooms:
                position = self._compute_position(room, rule)
                self._placements.append({
                    "device_type": rule.device_type,
                    "room_id": room.room_id,
                    "room_type": room.room_type,
                    "position": position,
                })
        
        return self._placements
    
    def _compute_position(self, room: Any, rule: PlacementRule) -> Tuple[float, float, float]:
        """Compute position within a room."""
        if rule.position_type == "center":
            return (room.center[0], rule.height, room.center[2])
        elif rule.position_type == "corner":
            return (room.bounds_min[0] + 0.5, rule.height, room.bounds_min[2] + 0.5)
        elif rule.position_type == "door":
            return (room.bounds_min[0], rule.height, room.center[2])
        else:  # wall
            return (room.center[0], rule.height, room.bounds_max[2] - 0.1)
    
    def _get_default_placements(self) -> List[Dict[str, Any]]:
        """Get default placements when no scene is loaded."""
        return [
            {"device_type": "motion_sensor", "position": (0, 2.5, 0)},
            {"device_type": "smart_door", "position": (3, 1.0, 0)},
            {"device_type": "contact_sensor", "position": (3, 1.0, 0)},
            {"device_type": "light_sensor", "position": (-2, 2.0, 2)},
        ]
    
    def apply_placements(self) -> int:
        """Apply computed placements to the environment."""
        if not self.environment:
            logger.warning("No environment configured")
            return 0
        
        count = 0
        for placement in self._placements:
            device = self._create_device(placement)
            if device:
                self.environment.register_device(device)
                count += 1
        
        logger.info(f"Placed {count} devices in environment")
        return count
    
    def _create_device(self, placement: Dict[str, Any]) -> Optional[IoTDevice]:
        """Create a device from placement info."""
        from vesper.devices import MotionSensor, ContactSensor, SmartDoor, LightSensor
        
        device_type = placement["device_type"]
        position = placement["position"]
        
        device_classes = {
            "motion_sensor": MotionSensor,
            "contact_sensor": ContactSensor,
            "smart_door": SmartDoor,
            "light_sensor": LightSensor,
        }
        
        cls = device_classes.get(device_type)
        if cls:
            return cls(location=position, event_bus=self.environment.event_bus)
        return None
