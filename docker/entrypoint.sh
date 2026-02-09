#!/bin/bash
# VESPER Virtual Device Entrypoint
# Runs QEMU with firmware and exposes serial port via TCP

set -e

echo "╔══════════════════════════════════════════╗"
echo "║  VESPER Virtual IoT Device Container     ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Device ID:   ${DEVICE_ID}               "
echo "║  Device Name: ${DEVICE_NAME}              "
echo "║  Machine:     ${QEMU_MACHINE}             "
echo "║  CPU:         ${QEMU_CPU}                 "
echo "║  Serial Port: ${SERIAL_PORT}              "
echo "╚══════════════════════════════════════════╝"

# Start QEMU with serial output over TCP
# -serial tcp::PORT,server=on,wait=off  makes QEMU listen on a TCP port for serial I/O
exec qemu-system-arm \
    -M "${QEMU_MACHINE}" \
    -cpu "${QEMU_CPU}" \
    -nographic \
    -kernel /firmware/sensor_firmware.elf \
    -serial "tcp::${SERIAL_PORT},server=on,wait=off"
