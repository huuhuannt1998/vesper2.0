"""
QEMU/Firmware integration module for Vesper.

Provides framework for running emulated IoT device firmware.
"""

from vesper.firmware.emulator import FirmwareEmulator, EmulatorConfig
from vesper.firmware.bridge import FirmwareBridge, BridgeConfig
from vesper.firmware.device_fw import DeviceFirmware, FirmwareState

__all__ = [
    "FirmwareEmulator",
    "EmulatorConfig",
    "FirmwareBridge",
    "BridgeConfig",
    "DeviceFirmware",
    "FirmwareState",
]
