import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import threading

class RGBDCaptureNode(Node):
    def __init__(self):
        super().__init__('rgbd_capture_node')
        
        # Initialize CvBridge for converting ROS Image to OpenCV
        self.bridge = CvBridge()
        
        # Create subscriptions for RGB and Depth topics
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Change topic to your camera's RGB topic
            self.rgb_callback,
            10
        )
        
        self.get_logger().info("RGBD Capture Node started")
        
        # Variables to store the images when received
        self.rgb_image = None
        self.depth_image = None
        
        # Create a thread for user input to capture images
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.start()

    def rgb_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Error while processing RGB image: {e}")

    
    def wait_for_input(self):
        while True:
            user_input = input("Type 'Y' to take a photo: ")
            if user_input == "Y":
                self.capture_and_save_images()

    def capture_and_save_images(self):
        if self.rgb_image is not None:
            # Save RGB image
            rgb_filename = 'rgb_image.png'
            cv2.imwrite(rgb_filename, self.rgb_image)
            self.get_logger().info(f"RGB Image saved to {rgb_filename}")
        
        else:
            self.get_logger().warn("RGB or Depth image not yet available. Please try again.")

def main(args=None):
    rclpy.init(args=args)
    node = RGBDCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
