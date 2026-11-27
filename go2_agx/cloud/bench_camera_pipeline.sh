#!/bin/bash
# GO2 (Jetson). Cost of the Odin camera video pipeline WITHOUT AWS. It runs the same GStreamer chain that media_gateway builds
# for source_type ros_compressed_image + encoder nvv4l2h264 + clock overlay (packages/gateways/media-gateway in ardia-robotics),
# fed with ONE real camera frame repeated at the stream rate, and prints the achieved fps and the CPU use. The WebRTC part
# (KVS signaling, needs AWS) is not included. It starts one throwaway container (odin-bench, removed at the end) and changes nothing else.
# Needs: the stack running (the camera stream on :8080), the image ardia/pulse-runtime:dev, the NVIDIA runtime.
#   ./bench_camera_pipeline.sh [seconds, default 30]
# Settings (environment): PULSE_IMAGE, W (1280) H (1040) FPS (10) KBPS (2000)
set -euo pipefail
SECS=${1:-30}
IMAGE=${PULSE_IMAGE:-ardia/pulse-runtime:dev}
W=${W:-1280}; H=${H:-1040}; FPS=${FPS:-10}; KBPS=${KBPS:-2000}
TMP=$(mktemp -d)
trap 'docker rm -f odin-bench >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT
DOCKER_GPU=(--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility)

echo "== 1. one real camera frame from http://127.0.0.1:8080/"
python3 - "$TMP/frame.jpg" <<'EOF'
import sys, urllib.request
data = b""
with urllib.request.urlopen("http://127.0.0.1:8080/", timeout=10) as r:
    while len(data) < 6_000_000:
        data += r.read(65536)
        a = data.find(b"\xff\xd8")
        if a >= 0:
            b = data.find(b"\xff\xd9", a + 2)
            if b >= 0 and b - a > 50_000:
                open(sys.argv[1], "wb").write(data[a:b + 2])
                print("   frame: %d bytes" % (b + 2 - a))
                sys.exit(0)
sys.exit("no complete JPEG frame found in the stream")
EOF

echo "== 2. GStreamer elements in $IMAGE"
for e in jpegdec nvvidconv nvv4l2h264enc textoverlay clockoverlay h264parse fpsdisplaysink; do
  if docker run --rm "${DOCKER_GPU[@]}" --entrypoint gst-inspect-1.0 "$IMAGE" "$e" >/dev/null 2>&1; then echo "   ok       $e"; else echo "   MISSING  $e"; MISSING=1; fi
done
[ -z "${MISSING:-}" ] || { echo "ERROR: elements missing: the pipeline cannot run in this image"; exit 1; }

cat > "$TMP/run.sh" <<EOF2
exec timeout -s INT $SECS gst-launch-1.0 -e -v \\
  multifilesrc location=/bench/frame.jpg loop=true caps="image/jpeg,framerate=$FPS/1" ! \\
  queue leaky=downstream max-size-buffers=1 ! \\
  jpegdec ! nvvidconv ! "video/x-raw(memory:NVMM),format=NV12,width=$W,height=$H,framerate=$FPS/1" ! \\
  nvvidconv ! video/x-raw,format=I420 ! \\
  textoverlay text="go2-000.odin" halignment=left valignment=top xpad=10 ypad=10 font-desc="Ubuntu Mono 10" ! \\
  clockoverlay halignment=right valignment=top xpad=10 ypad=10 time-format="%Y-%m-%d %H:%M:%S" font-desc="Ubuntu Mono 10" ! \\
  nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \\
  nvv4l2h264enc insert-sps-pps=true bitrate=$((KBPS * 1000)) control-rate=1 iframeinterval=$FPS idrinterval=$FPS preset-level=1 maxperf-enable=true ! \\
  h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au,profile=baseline ! \\
  fpsdisplaysink video-sink="fakesink sync=true" text-overlay=false
EOF2

echo "== 3. pipeline for $SECS s: ${W}x${H} at $FPS fps, $KBPS kbps"
docker rm -f odin-bench >/dev/null 2>&1 || true
docker run -d --name odin-bench --network host "${DOCKER_GPU[@]}" -v "$TMP:/bench:ro" --entrypoint bash "$IMAGE" /bench/run.sh >/dev/null
sleep $((SECS / 2))
echo "   CPU during the run (percent of one core; the Jetson has many):"
docker stats --no-stream --format "     {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" odin-bench odin-agx ons2-go2 odin-gpu 2>/dev/null || true
docker wait odin-bench >/dev/null 2>&1 || true
LOG=$(docker logs odin-bench 2>&1)

echo "== 4. result"
echo "$LOG" | grep -E "ERROR|WARNING: erroneous|Failed|failed" | head -5 | sed 's/^/   /' || true
LAST=$(echo "$LOG" | grep "last-message" | tail -1)
if [ -n "$LAST" ]; then
  echo "   $(echo "$LAST" | sed 's/.*last-message = //')"
  AVG=$(echo "$LAST" | sed -n 's/.*average: \([0-9.]*\).*/\1/p')
  if [ -n "$AVG" ] && awk -v a="$AVG" -v f="$FPS" 'BEGIN{exit !(a > f * 1.5)}'; then
    echo "   NOTE: average $AVG fps is far above $FPS: the source was not paced, so this is the MAXIMUM throughput of the chain"
  fi
else
  echo "   FAIL: no frame was rendered. Last log lines:"
  echo "$LOG" | tail -8 | sed 's/^/     /'
  exit 1
fi
