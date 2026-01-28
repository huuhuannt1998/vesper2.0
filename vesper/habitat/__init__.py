"""
Habitat-Sim integration module for Vesper.

Provides 3D simulation capabilities using Facebook's Habitat-Sim.
"""

from vesper.habitat.simulator import HabitatSimulator, SimulatorConfig
from vesper.habitat.scene import SceneManager, SceneConfig
from vesper.habitat.embodiment import EmbodiedAgent, EmbodimentConfig
from vesper.habitat.device_placement import DevicePlacer, PlacementConfig

__all__ = [
    "HabitatSimulator",
    "SimulatorConfig",
    "SceneManager",
    "SceneConfig",
    "EmbodiedAgent",
    "EmbodimentConfig",
    "DevicePlacer",
    "PlacementConfig",
]
