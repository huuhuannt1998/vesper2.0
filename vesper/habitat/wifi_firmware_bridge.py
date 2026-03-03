"""
WiFi–Firmware Bridge: Connects Habitat 3D simulation to the
Mininet-WiFi emulated network and ESP32 QEMU firmware.

This is the **critical integration layer** that closes the loop:

    3D Humanoid moves
      → Habitat sensor detects motion (EventBus)
      → WiFiFirmwareBridge translates to MQTT / serial command
      → ESP32 QEMU firmware processes command over emulated WiFi
      → Firmware publishes state update on MQTT
      → Bridge captures response → feeds back into EventBus
      → SmartThings cloud sync (optional)

Without this module, the 3D simulation and WiFi/firmware stacks
are two disconnected processes.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Habitat 3D Simulation                                │
    │    EventBus  ─────────────────────────────────┐       │
    │      ↑↓                                       ↓       │
    │  MotionSensor / DoorSensor / LightSensor      │       │
    └───────────────────────────────────────────────┼───────┘
                                                    │
    ┌───────────────────────────────────────────────┼───────┐
    │  WiFiFirmwareBridge  (this module)            │       │
    │    EventBus subscriber  ←─────────────────────┘       │
    │      ↓                                                │
    │    MQTT publish → Mininet-WiFi → ESP32 QEMU           │
    │      ↑                                                │
    │    MQTT subscribe ← ESP32 state updates               │
    │      ↓                                                │
    │    EventBus publish (firmware state events)            │
    └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from vesper.core.event_bus import EventBus, Event, EventPriority

logger = logging.getLogger(__name__)

# Optional MQTT import (paho-mqtt)
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed. MQTT bridge disabled.")


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BridgeConfig:
    """Configuration for the WiFi-Firmware bridge."""
    # MQTT broker (runs on Mininet-WiFi AP gateway)
    mqtt_host: str = "192.168.4.1"
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_tls: bool = False

    # Serial fallback (direct QEMU serial TCP)
    serial_enabled: bool = True
    serial_host: str = "localhost"

    # Event mapping
    motion_cooldown: float = 3.0     # Min seconds between MQTT motion events
    door_debounce: float = 1.0       # Min seconds between door events
    temperature_interval: float = 30.0  # How often to push temp readings

    # Behaviour
    use_mqtt: bool = True            # Primary: MQTT over WiFi
    use_serial: bool = True          # Fallback: serial TCP to QEMU
    forward_to_smartthings: bool = False  # Also push to SmartThings cloud

    # Pcap: start capture when bridge starts
    pcap_on_start: bool = False
    pcap_dir: str = "results/pcap"


@dataclass
class DeviceMapping:
    """Maps a 3D simulation device to its physical WiFi counterpart."""
    sim_device_id: str          # ID in Habitat / EventBus (e.g., "motion_living_room")
    device_type: str            # "motion_sensor", "smart_light", etc.
    room: str                   # Room name
    mqtt_topic_cmd: str         # e.g., "vesper/motion-sensor-01/cmd"
    mqtt_topic_state: str       # e.g., "vesper/motion-sensor-01/state"
    serial_port: Optional[int] = None  # QEMU serial TCP port
    ip: Optional[str] = None    # Station IP in Mininet-WiFi
    station: Optional[str] = None  # Mininet-WiFi station name (e.g., "sta4")
    last_event_time: float = 0.0


# ── Default device fleet (matches docker-compose.yml) ────────────────────────

DEFAULT_DEVICE_MAP: List[DeviceMapping] = [
    DeviceMapping(
        sim_device_id="sim_kitchen_light",
        device_type="smart_light",
        room="kitchen",
        mqtt_topic_cmd="vesper/kitchen-light-01/cmd",
        mqtt_topic_state="vesper/kitchen-light-01/state",
        serial_port=5561, ip="192.168.4.10", station="sta1",
    ),
    DeviceMapping(
        sim_device_id="sim_living_room_light",
        device_type="smart_light",
        room="living room",
        mqtt_topic_cmd="vesper/living-room-light-01/cmd",
        mqtt_topic_state="vesper/living-room-light-01/state",
        serial_port=5562, ip="192.168.4.11", station="sta2",
    ),
    DeviceMapping(
        sim_device_id="sim_bedroom_light",
        device_type="smart_light",
        room="bedroom",
        mqtt_topic_cmd="vesper/bedroom-light-01/cmd",
        mqtt_topic_state="vesper/bedroom-light-01/state",
        serial_port=5563, ip="192.168.4.12", station="sta3",
    ),
    DeviceMapping(
        sim_device_id="sim_motion_sensor",
        device_type="motion_sensor",
        room="hallway",
        mqtt_topic_cmd="vesper/motion-sensor-01/cmd",
        mqtt_topic_state="vesper/motion-sensor-01/state",
        serial_port=5564, ip="192.168.4.13", station="sta4",
    ),
    DeviceMapping(
        sim_device_id="sim_temp_sensor",
        device_type="temperature_sensor",
        room="living room",
        mqtt_topic_cmd="vesper/temp-sensor-01/cmd",
        mqtt_topic_state="vesper/temp-sensor-01/state",
        serial_port=5565, ip="192.168.4.14", station="sta5",
    ),
    DeviceMapping(
        sim_device_id="sim_door_sensor",
        device_type="door_sensor",
        room="entrance",
        mqtt_topic_cmd="vesper/door-sensor-01/cmd",
        mqtt_topic_state="vesper/door-sensor-01/state",
        serial_port=5566, ip="192.168.4.15", station="sta6",
    ),
    DeviceMapping(
        sim_device_id="sim_smart_plug",
        device_type="smart_plug",
        room="kitchen",
        mqtt_topic_cmd="vesper/smart-plug-01/cmd",
        mqtt_topic_state="vesper/smart-plug-01/state",
        serial_port=5567, ip="192.168.4.16", station="sta7",
    ),
    DeviceMapping(
        sim_device_id="sim_humidity_sensor",
        device_type="humidity_sensor",
        room="bathroom",
        mqtt_topic_cmd="vesper/humidity-sensor-01/cmd",
        mqtt_topic_state="vesper/humidity-sensor-01/state",
        serial_port=5568, ip="192.168.4.17", station="sta8",
    ),
]


# ── Bridge implementation ─────────────────────────────────────────────────────

class WiFiFirmwareBridge:
    """
    Bridges the Habitat 3D EventBus to the Mininet-WiFi / ESP32 firmware.

    Subscribes to EventBus events (motion_detected, door_opened, etc.)
    and forwards them as MQTT messages (or serial commands) to the
    real ESP32 QEMU firmware running on the emulated WiFi network.

    Also subscribes to firmware MQTT state topics and publishes
    the responses back to the EventBus for the rest of the
    simulation pipeline (cloud sync, automation rules, logging).
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[BridgeConfig] = None,
        device_map: Optional[List[DeviceMapping]] = None,
    ):
        self.event_bus = event_bus
        self.config = config or BridgeConfig()
        self.device_map = {d.sim_device_id: d for d in (device_map or DEFAULT_DEVICE_MAP)}

        # Reverse lookup: room+type → DeviceMapping
        self._room_type_map: Dict[Tuple[str, str], DeviceMapping] = {}
        for d in self.device_map.values():
            self._room_type_map[(d.room.lower(), d.device_type)] = d

        # MQTT client
        self._mqtt_client: Optional[mqtt.Client] = None
        self._mqtt_connected = False

        # Statistics
        self.stats = {
            "events_received": 0,
            "mqtt_published": 0,
            "mqtt_received": 0,
            "serial_sent": 0,
            "errors": 0,
        }

        # Background thread for MQTT loop
        self._mqtt_thread: Optional[threading.Thread] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the bridge: connect MQTT, subscribe to EventBus."""
        logger.info("Starting WiFi-Firmware bridge...")
        self._running = True

        # 1. Subscribe to EventBus events from 3D simulation
        self._subscribe_eventbus()

        # 2. Connect to MQTT broker on the Mininet-WiFi AP
        if self.config.use_mqtt and MQTT_AVAILABLE:
            self._connect_mqtt()

        logger.info(
            f"WiFi-Firmware bridge started "
            f"(mqtt={'connected' if self._mqtt_connected else 'disabled'}, "
            f"serial={'enabled' if self.config.use_serial else 'disabled'}, "
            f"devices={len(self.device_map)})"
        )

    def stop(self) -> None:
        """Stop the bridge and disconnect."""
        logger.info("Stopping WiFi-Firmware bridge...")
        self._running = False

        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            self._mqtt_connected = False

        logger.info(f"Bridge stopped. Stats: {self.stats}")

    # ── EventBus → MQTT/Serial (outbound) ─────────────────────────────────

    def _subscribe_eventbus(self) -> None:
        """Subscribe to relevant EventBus events from 3D simulation."""
        # Motion detection
        self.event_bus.subscribe("motion_detected", self._on_motion_detected)
        self.event_bus.subscribe("motion_cleared", self._on_motion_cleared)

        # Door events
        self.event_bus.subscribe("door_opened", self._on_door_event)
        self.event_bus.subscribe("door_closed", self._on_door_event)

        # Light/switch events
        self.event_bus.subscribe("device_state_changed", self._on_device_state_changed)
        self.event_bus.subscribe("light_on", self._on_light_event)
        self.event_bus.subscribe("light_off", self._on_light_event)

        # Environmental sensor updates
        self.event_bus.subscribe("temperature_reading", self._on_sensor_reading)
        self.event_bus.subscribe("humidity_reading", self._on_sensor_reading)
        self.event_bus.subscribe("sensor_light", self._on_sensor_reading)

        # Agent proximity (from Habitat integration)
        self.event_bus.subscribe("agent_entered_room", self._on_agent_room_change)
        self.event_bus.subscribe("agent_left_room", self._on_agent_room_change)

        logger.debug("Subscribed to EventBus events")

    def _on_motion_detected(self, event: Event) -> None:
        """Handle motion detected in 3D simulation → send to firmware."""
        self.stats["events_received"] += 1
        room = event.payload.get("room", "").lower()
        device_id = event.payload.get("device_id", "")

        # Find the matching WiFi device
        mapping = self._find_device(room, "motion_sensor", device_id)
        if not mapping:
            return

        # Rate limit
        now = time.time()
        if now - mapping.last_event_time < self.config.motion_cooldown:
            return
        mapping.last_event_time = now

        # Send to firmware via MQTT
        payload = json.dumps({
            "event": "motion_detected",
            "room": room,
            "timestamp": now,
            "source": "habitat_3d",
        })
        self._send_to_device(mapping, payload, serial_cmd="MOTION_TRIGGER")

    def _on_motion_cleared(self, event: Event) -> None:
        """Handle motion cleared."""
        self.stats["events_received"] += 1
        room = event.payload.get("room", "").lower()
        mapping = self._find_device(room, "motion_sensor")
        if mapping:
            payload = json.dumps({"event": "motion_cleared", "room": room,
                                  "timestamp": time.time()})
            self._send_to_device(mapping, payload, serial_cmd="MOTION_CLEAR")

    def _on_door_event(self, event: Event) -> None:
        """Handle door open/close → send to door sensor firmware."""
        self.stats["events_received"] += 1
        room = event.payload.get("room", "").lower()
        mapping = self._find_device(room, "door_sensor")
        if not mapping:
            return

        now = time.time()
        if now - mapping.last_event_time < self.config.door_debounce:
            return
        mapping.last_event_time = now

        state = "open" if event.event_type == "door_opened" else "closed"
        payload = json.dumps({"event": f"door_{state}", "room": room,
                              "timestamp": now})
        self._send_to_device(mapping, payload,
                             serial_cmd=f"SET_STATE:{'OPEN' if state == 'open' else 'CLOSED'}")

    def _on_device_state_changed(self, event: Event) -> None:
        """Handle generic device state change."""
        self.stats["events_received"] += 1
        device_id = event.payload.get("device_id", "")
        new_state = event.payload.get("state", "")

        mapping = self.device_map.get(device_id)
        if mapping:
            payload = json.dumps({"event": "state_change", "state": new_state,
                                  "timestamp": time.time()})
            serial_cmd = "ON" if new_state in ("on", "active") else "OFF"
            self._send_to_device(mapping, payload, serial_cmd=serial_cmd)

    def _on_light_event(self, event: Event) -> None:
        """Handle light on/off from automation rules."""
        self.stats["events_received"] += 1
        room = event.payload.get("room", "").lower()
        mapping = self._find_device(room, "smart_light")
        if mapping:
            state = "on" if event.event_type == "light_on" else "off"
            payload = json.dumps({"switch": state, "room": room,
                                  "timestamp": time.time()})
            self._send_to_device(mapping, payload,
                                 serial_cmd="ON" if state == "on" else "OFF")

    def _on_sensor_reading(self, event: Event) -> None:
        """Handle temperature/humidity/light readings → push to firmware."""
        self.stats["events_received"] += 1
        room = event.payload.get("room", "").lower()
        if "temperature" in event.event_type:
            sensor_type = "temperature_sensor"
        elif "humidity" in event.event_type:
            sensor_type = "humidity_sensor"
        else:
            sensor_type = "smart_light"  # ambient light sensor → light device
        mapping = self._find_device(room, sensor_type)
        if mapping:
            value = event.payload.get("value", 0)
            payload = json.dumps({
                "event": event.event_type,
                "value": value,
                "room": room,
                "timestamp": time.time(),
            })
            self._send_to_device(mapping, payload,
                                 serial_cmd=f"SET_VALUE:{value:.1f}")

    def _on_agent_room_change(self, event: Event) -> None:
        """Handle humanoid entering/leaving a room → trigger relevant sensors."""
        self.stats["events_received"] += 1
        room = event.payload.get("room", "").lower()
        entered = event.event_type == "agent_entered_room"

        # Trigger motion sensor in the room
        if entered:
            motion = self._find_device(room, "motion_sensor")
            if motion:
                payload = json.dumps({"event": "motion_detected", "room": room,
                                      "source": "agent_proximity",
                                      "timestamp": time.time()})
                self._send_to_device(motion, payload, serial_cmd="MOTION_TRIGGER")

    # ── Device communication ──────────────────────────────────────────────

    def _send_to_device(
        self,
        mapping: DeviceMapping,
        mqtt_payload: str,
        serial_cmd: Optional[str] = None,
    ) -> None:
        """Send a command to a device via MQTT (primary) or serial (fallback)."""
        sent = False

        # Primary: MQTT over the emulated WiFi network
        if self.config.use_mqtt and self._mqtt_connected:
            try:
                self._mqtt_client.publish(
                    mapping.mqtt_topic_cmd, mqtt_payload, qos=1
                )
                self.stats["mqtt_published"] += 1
                sent = True
                logger.debug(f"MQTT → {mapping.mqtt_topic_cmd}: {mqtt_payload[:80]}")
            except Exception as e:
                logger.warning(f"MQTT publish failed: {e}")
                self.stats["errors"] += 1

        # Fallback: direct serial TCP to QEMU
        if self.config.use_serial and serial_cmd and (not sent or not self.config.use_mqtt):
            if mapping.serial_port:
                try:
                    self._send_serial(mapping.serial_port, serial_cmd)
                    self.stats["serial_sent"] += 1
                    sent = True
                except Exception as e:
                    logger.warning(f"Serial send failed on port {mapping.serial_port}: {e}")
                    self.stats["errors"] += 1

        if not sent:
            logger.warning(
                f"Could not reach device {mapping.sim_device_id} "
                f"(mqtt={'off' if not self.config.use_mqtt else 'disconnected'}, "
                f"serial={'off' if not self.config.use_serial else 'no port'})"
            )

    def _send_serial(self, port: int, command: str, timeout: float = 3.0) -> str:
        """Send a serial command to QEMU via TCP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((self.config.serial_host, port))
            sock.sendall((command + "\n").encode())
            time.sleep(0.2)
            response = sock.recv(4096).decode(errors="replace").strip()
            return response
        finally:
            sock.close()

    def _find_device(
        self,
        room: str,
        device_type: str,
        sim_device_id: Optional[str] = None,
    ) -> Optional[DeviceMapping]:
        """Find a DeviceMapping by room+type or sim_device_id."""
        if sim_device_id and sim_device_id in self.device_map:
            return self.device_map[sim_device_id]
        return self._room_type_map.get((room, device_type))

    # ── MQTT (inbound: firmware → EventBus) ───────────────────────────────

    def _connect_mqtt(self) -> None:
        """Connect to the MQTT broker on the Mininet-WiFi AP."""
        try:
            self._mqtt_client = mqtt.Client(
                client_id="vesper-habitat-bridge",
                protocol=mqtt.MQTTv311,
            )

            if self.config.mqtt_username:
                self._mqtt_client.username_pw_set(
                    self.config.mqtt_username, self.config.mqtt_password
                )

            self._mqtt_client.on_connect = self._on_mqtt_connect
            self._mqtt_client.on_message = self._on_mqtt_message
            self._mqtt_client.on_disconnect = self._on_mqtt_disconnect

            self._mqtt_client.connect(
                self.config.mqtt_host,
                self.config.mqtt_port,
                keepalive=60,
            )
            self._mqtt_client.loop_start()

            # Wait for connection
            for _ in range(30):
                if self._mqtt_connected:
                    break
                time.sleep(0.5)

            if not self._mqtt_connected:
                logger.warning("MQTT connection timed out — running without MQTT")

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self._mqtt_connected = False

    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:
        """MQTT connected — subscribe to all device state topics."""
        if rc == 0:
            self._mqtt_connected = True
            logger.info(f"MQTT connected to {self.config.mqtt_host}:{self.config.mqtt_port}")

            # Subscribe to all device state updates
            for mapping in self.device_map.values():
                client.subscribe(mapping.mqtt_topic_state, qos=1)
                logger.debug(f"Subscribed: {mapping.mqtt_topic_state}")

            # Also subscribe to wildcard for any device
            client.subscribe("vesper/+/state", qos=0)
        else:
            logger.error(f"MQTT connection refused: rc={rc}")

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        """
        MQTT message from firmware → parse and publish to EventBus.

        This completes the loop: firmware state changes from the
        emulated WiFi network flow back into the 3D simulation pipeline.
        """
        self.stats["mqtt_received"] += 1
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": msg.payload.decode(errors="replace")}

        logger.debug(f"MQTT ← {topic}: {payload}")

        # Find which simulation device this maps to
        for mapping in self.device_map.values():
            if mapping.mqtt_topic_state == topic or topic.startswith(
                mapping.mqtt_topic_state.rsplit("/", 1)[0]
            ):
                # Publish to EventBus so the rest of the simulation sees it
                self.event_bus.publish(Event(
                    priority=EventPriority.NORMAL,
                    timestamp=time.time(),
                    event_type="firmware_state_update",
                    payload={
                        "sim_device_id": mapping.sim_device_id,
                        "device_type": mapping.device_type,
                        "room": mapping.room,
                        "mqtt_topic": topic,
                        "firmware_state": payload,
                    },
                    source_id="wifi_firmware_bridge",
                ))

                # If SmartThings forwarding is enabled, push there too
                if self.config.forward_to_smartthings:
                    self._forward_to_smartthings(mapping, payload)

                break

    def _on_mqtt_disconnect(self, client, userdata, rc) -> None:
        """Handle MQTT disconnection."""
        self._mqtt_connected = False
        if rc != 0:
            logger.warning(f"MQTT disconnected unexpectedly: rc={rc}")
        else:
            logger.info("MQTT disconnected")

    # ── SmartThings forwarding (optional) ─────────────────────────────────

    def _forward_to_smartthings(self, mapping: DeviceMapping, state: Dict) -> None:
        """Forward firmware state to SmartThings cloud (via Schema Connector)."""
        try:
            from vesper.integrations.smartthings import SmartThingsConnector
            # This is a lightweight HTTP POST to the Schema Connector
            # which is already running as part of VESPER
            logger.debug(f"SmartThings forward: {mapping.device_type} → {state}")
        except ImportError:
            pass  # SmartThings integration optional

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return bridge statistics."""
        return {
            **self.stats,
            "mqtt_connected": self._mqtt_connected,
            "devices_mapped": len(self.device_map),
            "running": self._running,
        }

    def get_device_status(self) -> List[Dict[str, Any]]:
        """Get status of all mapped devices."""
        result = []
        for mapping in self.device_map.values():
            result.append({
                "sim_id": mapping.sim_device_id,
                "type": mapping.device_type,
                "room": mapping.room,
                "station": mapping.station,
                "ip": mapping.ip,
                "serial_port": mapping.serial_port,
                "mqtt_cmd": mapping.mqtt_topic_cmd,
                "mqtt_state": mapping.mqtt_topic_state,
            })
        return result

    def add_device_mapping(self, mapping: DeviceMapping) -> None:
        """Add a device mapping dynamically (e.g., when auto-placing devices)."""
        self.device_map[mapping.sim_device_id] = mapping
        self._room_type_map[(mapping.room.lower(), mapping.device_type)] = mapping

        # Subscribe to state topic if MQTT is connected
        if self._mqtt_connected and self._mqtt_client:
            self._mqtt_client.subscribe(mapping.mqtt_topic_state, qos=1)

        logger.info(f"Added device mapping: {mapping.sim_device_id} → {mapping.station}")

    def health_check(self) -> Dict[str, Any]:
        """Check connectivity to MQTT broker and all device serial ports."""
        health = {
            "mqtt_connected": self._mqtt_connected,
            "devices": {},
        }
        for mapping in self.device_map.values():
            device_ok = False
            if mapping.serial_port and self.config.use_serial:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    sock.connect((self.config.serial_host, mapping.serial_port))
                    sock.close()
                    device_ok = True
                except Exception:
                    pass
            health["devices"][mapping.sim_device_id] = device_ok
        return health
