#!/usr/bin/env python3
# AGX ORIN (runs inside the odin-agx container). Serves the Odin1 color camera over HTTP, as an MJPEG stream with
# the original JPEG frames (1600x1296, full quality, no re-encoding), straight to the laptop over Wi-Fi.
# ROS 2 / DDS over Wi-Fi loses the ~0.7 MB images (tested 9 Nov: 0.1 fps); TCP does not.
#   http://<AGX>:8080/            page with the live image (any browser)
#   http://<AGX>:8080/stream.mjpg MJPEG stream (view_camera.sh on the laptop)
#   http://<AGX>:8080/snapshot.jpg last frame
import threading, time, rospy
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from sensor_msgs.msg import CompressedImage

PORT = 8080
cond = threading.Condition()
frame = {'data': None, 'n': 0}

def cb(m):
    with cond:
        frame['data'] = bytes(m.data); frame['n'] += 1; cond.notify_all()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith('/stream.mjpg'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache'); self.end_headers()
            last = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: frame['n'] != last, timeout=5)
                        if frame['n'] == last: continue
                        data, last = frame['data'], frame['n']
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(data))
                    self.wfile.write(data); self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith('/snapshot.jpg') and frame['data']:
            self.send_response(200); self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(frame['data']))); self.end_headers()
            self.wfile.write(frame['data'])
        else:
            page = b'<html><body style="margin:0;background:#000"><img src="/stream.mjpg" style="width:100%"></body></html>'
            self.send_response(200); self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(page))); self.end_headers(); self.wfile.write(page)

rospy.init_node('odin1_cam_stream', anonymous=True, disable_signals=True)
rospy.Subscriber('/odin1/image/compressed', CompressedImage, cb, queue_size=1, buff_size=4 * 1024 * 1024)
print('Odin1 camera stream on port %d' % PORT, flush=True)
ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
