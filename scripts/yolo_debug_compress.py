#!/usr/bin/env python3
# OUR helper (not ONS2 upstream): their yolo_detector publishes /yolo_debug_image as a RAW 1600x1296 image (~6 MB),
# too heavy for Foxglove over Wi-Fi (it crashed foxglove_bridge). This republishes it as a small JPEG:
#   /yolo_debug_image  ->  /yolo_debug_image/compressed  (800 px wide, quality 70)
import cv2, rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
rospy.init_node("yolo_debug_compress")
bridge = CvBridge()
pub = rospy.Publisher("/yolo_debug_image/compressed", CompressedImage, queue_size=1)
def cb(m):
    img = bridge.imgmsg_to_cv2(m, "bgr8")
    h, w = img.shape[:2]
    img = cv2.resize(img, (800, int(h * 800 / w)))
    out = CompressedImage(header=m.header, format="jpeg")
    out.data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    pub.publish(out)
rospy.Subscriber("/yolo_debug_image", Image, cb, queue_size=1, buff_size=2**24)
rospy.spin()
