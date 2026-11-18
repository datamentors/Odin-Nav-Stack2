#!/bin/sh
# Odin1 on the AGX Orin: keep USB 3 links awake. With USB autosuspend + USB3 link power management (U1/U2)
# the Odin1 (high-bandwidth, Realtek hub 0bda:0420 = bus 2-3) dropped with "Cannot enable. Maybe the USB cable is bad?".
echo -1 > /sys/module/usbcore/parameters/autosuspend
for d in /sys/bus/usb/devices/usb*/power/control /sys/bus/usb/devices/*-*/power/control; do [ -w "$d" ] && echo on > "$d"; done
for p in /sys/bus/usb/devices/*/*:1.0/*-port*/usb3_lpm_permit; do [ -w "$p" ] && echo 0 > "$p"; done
exit 0
