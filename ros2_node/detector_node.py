"""ROS 2 node: subscribe to /image_raw, run YOLO, publish /detections.

Needs ROS 2 (Humble or later) with cv_bridge and vision_msgs installed.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.declare_parameter("model", "best.onnx")
        self.model = YOLO(self.get_parameter("model").value)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/image_raw", self.callback, 10)
        self.pub = self.create_publisher(Detection2DArray, "/detections", 10)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        res = self.model(frame, verbose=False)[0]
        out = Detection2DArray()
        out.header = msg.header
        for box, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy(),
                                  res.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in box)
            d = Detection2D()
            d.header = msg.header
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            try:                                   # Iron and later
                d.bbox.center.position.x, d.bbox.center.position.y = cx, cy
            except AttributeError:                 # Humble
                d.bbox.center.x, d.bbox.center.y = cx, cy
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cls))
            hyp.hypothesis.score = float(conf)
            d.results.append(hyp)
            out.detections.append(d)
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DetectorNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
