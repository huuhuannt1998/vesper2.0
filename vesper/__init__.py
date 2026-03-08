"""
Vesper: WiFi-Faithful IoT Security Testbed.

A platform for simulating IoT devices over real WiFi (mac80211_hwsim),
with LLM-controlled agents, Matter device support via Home Assistant,
hub-based traffic routing, and a real-time monitoring dashboard.

Modules:
- hub: Centralized device routing (VirtualHub, PhysicalHub)
- matter: Matter/CHIP device integration via Home Assistant
- dashboard: Real-time Web UI for monitoring & control
- habitat: 3D environment simulation (Habitat 3.0)
- simulation: Time management, task system, event streaming
- integrations: External platform integrations (SmartThings)
"""

__version__ = "3.0.0"

from vesper.config import Config, load_config
from vesper.engine import VesperEngine

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "VesperEngine",
]
