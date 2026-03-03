"""
Network module for Vesper IoT communication.

Provides transport abstractions and simulated network capabilities.
"""

from vesper.network.transport import (
    Transport,
    TransportState,
    LocalTransport,
    SimulatedTransport,
)
from vesper.network.router import MessageRouter
from vesper.network.broker import MessageBroker

# MQTT support (optional - requires paho-mqtt)
try:
    from vesper.network.mqtt import (
        MQTTTransport,
        MQTTConfig,
        MQTTEventBridge,
        QoS,
    )
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    MQTTTransport = None
    MQTTConfig = None
    MQTTEventBridge = None
    QoS = None

# WiFi emulator (requires Docker + Mininet-WiFi)
try:
    from vesper.network.wifi_emulator import (
        WiFiEmulator,
        WiFiConfig,
        DeviceConfig,
        DeviceType,
    )
    WIFI_AVAILABLE = True
except ImportError:
    WIFI_AVAILABLE = False
    WiFiEmulator = None
    WiFiConfig = None
    DeviceConfig = None
    DeviceType = None

__all__ = [
    "Transport",
    "TransportState",
    "LocalTransport",
    "SimulatedTransport",
    "MessageRouter",
    "MessageBroker",
    # MQTT
    "MQTTTransport",
    "MQTTConfig",
    "MQTTEventBridge",
    "QoS",
    "MQTT_AVAILABLE",
    # WiFi emulator
    "WiFiEmulator",
    "WiFiConfig",
    "DeviceConfig",
    "DeviceType",
    "WIFI_AVAILABLE",
]
