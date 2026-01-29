"""
IoT Device models for Vesper.
"""

from vesper.devices.base import IoTDevice
from vesper.devices.motion_sensor import MotionSensor
from vesper.devices.contact_sensor import ContactSensor
from vesper.devices.smart_door import SmartDoor
from vesper.devices.light_sensor import LightSensor
from vesper.devices.manager import DeviceManager, DeviceManagerConfig, create_smart_home_setup

__all__ = [
    "IoTDevice",
    "MotionSensor",
    "ContactSensor",
    "SmartDoor",
    "LightSensor",
    "DeviceManager",
    "DeviceManagerConfig",
    "create_smart_home_setup",
]
