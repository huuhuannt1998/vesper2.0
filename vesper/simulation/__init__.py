"""
Simulation module for autonomous daily life simulation.

Provides:
- Time management with real-world synchronization
- Task system for humanoid activities
- Task generation with LLM support
- Task database for history and analytics
- Event stream for coordination
"""

from vesper.simulation.time_manager import (
    TimeManager,
    TimeConfig,
    TimeOfDay,
    DayOfWeek,
    ScheduledEvent,
)

from vesper.simulation.task_system import (
    Task,
    TaskAction,
    TaskCategory,
    TaskExecutor,
    TaskFactory,
    TaskLocation,
    TaskPriority,
    TaskStatus,
    DailySchedule,
)

from vesper.simulation.task_generator import (
    TaskGenerator,
    HumanoidPersona,
    RoomInfo,
)

from vesper.simulation.task_database import TaskDatabase

from vesper.simulation.event_stream import (
    Event,
    EventType,
    EventStream,
    EventLogger,
    SimulationCoordinator,
)

from vesper.simulation.autonomous_simulation import AutonomousSimulation

# Re-export the top-level Simulation runner class.
# The class lives in vesper/simulation.py but that file is shadowed by this
# package directory.  We load it by explicit file path to avoid circular imports.
import importlib.util as _ilu
import pathlib as _pl

_sim_file = _pl.Path(__file__).resolve().parent.parent / "simulation.py"
_spec = _ilu.spec_from_file_location("vesper._sim_runner", str(_sim_file))
_sim_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
# Temporarily prevent re-entry: register the half-loaded module
import sys as _sys
_sys.modules["vesper._sim_runner"] = _sim_mod
_spec.loader.exec_module(_sim_mod)  # type: ignore[union-attr]

Simulation = _sim_mod.Simulation
SimulationStats = _sim_mod.SimulationStats

__all__ = [
    # Time management
    "TimeManager",
    "TimeConfig",
    "TimeOfDay",
    "DayOfWeek",
    "ScheduledEvent",
    # Task system
    "Task",
    "TaskAction",
    "TaskCategory",
    "TaskExecutor",
    "TaskFactory",
    "TaskLocation",
    "TaskPriority",
    "TaskStatus",
    "DailySchedule",
    # Task generation
    "TaskGenerator",
    "HumanoidPersona",
    "RoomInfo",
    # Database
    "TaskDatabase",
    # Event stream
    "Event",
    "EventType",
    "EventStream",
    "EventLogger",
    "SimulationCoordinator",
    # Autonomous simulation
    "AutonomousSimulation",
    # Top-level simulation runner
    "Simulation",
    "SimulationStats",
]
