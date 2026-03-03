"""
Main simulation runner for Vesper.

Integrates all components into a unified simulation loop:
  - Habitat-Sim for 3D environment + humanoid navigation
  - IoT devices and sensors (in-process)
  - WiFi-Firmware bridge (EventBus → Mininet-WiFi → ESP32 QEMU)
  - LLM-controlled agents
  - Event bus for communication
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vesper.config import Config, load_config
from vesper.core.event_bus import EventBus
from vesper.core.environment import Environment
from vesper.agents.controller import AgentController
from vesper.habitat.simulator import create_simulator, SimulatorConfig

logger = logging.getLogger(__name__)


@dataclass
class SimulationStats:
    """Simulation statistics."""
    ticks: int = 0
    elapsed_time: float = 0.0
    avg_tick_time: float = 0.0
    firmware_events: int = 0
    mqtt_messages: int = 0


class Simulation:
    """
    Main simulation runner.
    
    Coordinates:
    - Habitat-Sim for 3D environment
    - IoT devices and sensors
    - WiFi-Firmware bridge (Mininet-WiFi ↔ ESP32 QEMU)
    - LLM-controlled agents
    - Event bus for communication
    
    Example:
        sim = Simulation()
        sim.initialize(wifi=True)
        sim.run(duration=60.0)  # Run for 60 seconds
        sim.close()
    """
    
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = config or Config()
        
        self.event_bus = EventBus(max_queue_size=self.config.event_bus.max_queue_size)
        self.environment = Environment(event_bus=self.event_bus)
        self.agent_controller = AgentController(event_bus=self.event_bus)
        self.simulator = None
        
        # WiFi / firmware integration (optional)
        self._wifi_emulator = None
        self._wifi_bridge = None
        self._wifi_enabled = False
        
        self._running = False
        self._stats = SimulationStats()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def stats(self) -> SimulationStats:
        return self._stats
    
    def initialize(self, use_mock_sim: bool = True, wifi: bool = False) -> bool:
        """
        Initialize all simulation components.
        
        Args:
            use_mock_sim: Use mock Habitat simulator (no GPU needed)
            wifi: Start Mininet-WiFi emulated network + ESP32 firmware bridge.
                  When True, 3D sensor events are forwarded to real QEMU ESP32
                  firmware over the emulated 802.11 WiFi network.
        """
        logger.info("Initializing Vesper simulation...")
        
        # Create simulator
        sim_config = SimulatorConfig(
            scene_path=self.config.environment.scene,
            render_mode="headless" if self.config.simulation.headless else "window",
        )
        self.simulator = create_simulator(sim_config, use_mock=use_mock_sim)
        self.simulator.initialize()
        
        # Set up devices from config
        self._setup_devices()
        
        # Optionally start WiFi/firmware integration
        if wifi:
            self._setup_wifi_bridge()
        
        logger.info("Simulation initialized")
        return True
    
    def _setup_wifi_bridge(self) -> None:
        """
        Start the Mininet-WiFi emulated network and connect the
        EventBus ↔ MQTT ↔ ESP32 firmware bridge.
        """
        try:
            from vesper.network.wifi_emulator import WiFiEmulator
            from vesper.habitat.wifi_firmware_bridge import WiFiFirmwareBridge, BridgeConfig

            # 1. Start the WiFi emulator (docker compose up)
            logger.info("Starting Mininet-WiFi emulated network...")
            self._wifi_emulator = WiFiEmulator()
            self._wifi_emulator.start(build=False, detach=True)
            self._wifi_emulator.wait_ready(timeout=120)
            logger.info("WiFi topology ready")

            # 2. Create the EventBus ↔ MQTT bridge
            bridge_config = BridgeConfig(
                mqtt_host=self._wifi_emulator.wifi.gateway_ip,
                mqtt_port=self._wifi_emulator.wifi.mqtt_port,
            )
            self._wifi_bridge = WiFiFirmwareBridge(
                event_bus=self.event_bus,
                config=bridge_config,
            )
            self._wifi_bridge.start()
            self._wifi_enabled = True

            logger.info("WiFi-Firmware bridge active — 3D events → MQTT → ESP32")
        except ImportError as e:
            logger.warning(f"WiFi bridge unavailable (missing dependency): {e}")
        except Exception as e:
            logger.error(f"Failed to start WiFi bridge: {e}")
            # Simulation continues without WiFi — graceful degradation
        
        logger.info("Simulation initialized")
        return True
    
    def _setup_devices(self) -> None:
        """Set up IoT devices from configuration."""
        from vesper.devices import MotionSensor, ContactSensor, SmartDoor, LightSensor
        
        device_configs = [
            ("motion_sensor", self.config.devices.motion_sensor, MotionSensor),
            ("contact_sensor", self.config.devices.contact_sensor, ContactSensor),
            ("smart_door", self.config.devices.smart_door, SmartDoor),
            ("light_sensor", self.config.devices.light_sensor, LightSensor),
        ]
        
        for name, dev_cfg, cls in device_configs:
            if dev_cfg.enabled:
                device = cls(event_bus=self.event_bus)
                self.environment.register_device(device)
                logger.debug(f"Registered {name}: {device.device_id}")
    
    def step(self, dt: Optional[float] = None) -> None:
        """Execute one simulation tick."""
        dt = dt or (1.0 / self.config.simulation.tick_rate)
        start = time.time()
        
        # Update simulator
        if self.simulator:
            self.simulator.step()
        
        # Update environment devices
        self.environment.tick(dt)
        
        # Update agents
        self.agent_controller.update(dt, self.environment)
        
        # Process events
        self.event_bus.process_events()
        
        self._stats.ticks += 1
        tick_time = time.time() - start
        self._stats.elapsed_time += tick_time
        self._stats.avg_tick_time = self._stats.elapsed_time / self._stats.ticks
    
    def run(self, duration: float = 10.0) -> None:
        """Run simulation for specified duration."""
        logger.info(f"Running simulation for {duration}s...")
        self._running = True
        
        dt = 1.0 / self.config.simulation.tick_rate
        end_time = time.time() + duration
        
        while self._running and time.time() < end_time:
            self.step(dt)
            time.sleep(max(0, dt - self._stats.avg_tick_time))
        
        self._running = False
        logger.info(f"Simulation complete: {self._stats.ticks} ticks")
    
    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
    
    def close(self) -> None:
        """Clean up all resources."""
        self.stop()

        # Stop WiFi bridge and emulator
        if self._wifi_bridge:
            try:
                self._wifi_bridge.stop()
                logger.info(f"WiFi bridge stats: {self._wifi_bridge.get_stats()}")
            except Exception as e:
                logger.warning(f"Error stopping WiFi bridge: {e}")

        if self._wifi_emulator:
            try:
                self._wifi_emulator.stop()
            except Exception as e:
                logger.warning(f"Error stopping WiFi emulator: {e}")

        self.agent_controller.destroy_all()
        if self.simulator:
            self.simulator.close()
        logger.info("Simulation closed")
    
    def __enter__(self) -> "Simulation":
        self.initialize()
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
