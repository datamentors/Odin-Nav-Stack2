#!/bin/bash
# AGX ORIN. Recovers the Odin1 when the kernel says "usb 2-3-port3: Cannot enable. Maybe the USB cable is bad?"
# and the Odin1 does not appear in lsusb: resets the internal USB 3.0 hub (2-3, Realtek 0bda:0420) where it is plugged.
#   ~/odin_test/fix_odin_usb.sh        (asks for your sudo password)
# Worked on 16 Nov. If it does not come back after this, the Odin1 is hung: remove its power for 10 s.
present() { for d in /sys/bus/usb/devices/*; do [ "$(cat $d/idVendor 2>/dev/null)" = 2207 ] && { echo "Odin1 OK on $(basename $d), speed=$(cat $d/speed) Mb/s"; return 0; }; done; return 1; }
present && exit 0
H=/sys/bus/usb/devices/2-3
[ -e $H/authorized ] || { echo "Hub 2-3 not found"; exit 1; }
echo "Odin1 missing: resetting the USB hub 2-3..."
echo 0 | sudo tee $H/authorized >/dev/null; sleep 4
echo 1 | sudo tee $H/authorized >/dev/null
for i in $(seq 1 10); do sleep 2; present && break; done
# after a hub reset the autosuspend/LPM settings go back to default: apply them again
[ -x /usr/local/bin/odin_usb_tuning.sh ] && sudo /usr/local/bin/odin_usb_tuning.sh >/dev/null 2>&1
present || { echo "Still missing: unplug the Odin1 power (or its USB cable) for 10 s, plug it back, and run this again."; exit 1; }
echo "Now run: ~/odin_test/start_odin_agx.sh go2"
