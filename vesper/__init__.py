"""
Vesper: Habitat 3.0 + IoT Interactive Simulation Testbed.

A framework for simulating IoT devices in 3D environments with
LLM-controlled agents and network protocol simulation.
"""

__version__ = "0.1.0"

from vesper.config import Config, load_config

__all__ = [
    "__version__",
    "Config",
    "load_config",
]
