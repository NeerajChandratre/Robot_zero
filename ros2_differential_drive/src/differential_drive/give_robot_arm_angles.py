import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
import numpy as np
from tf2_ros.transform_listener import TransformListener
import math
 
class user_inpt_publshr(Node):
    def __init__(self):
        super().__init__('user_input_publisher')
        self.publisher = self.create_publisher(String,'robot_arm_angles_here',10)
        self.get_logger().info("publshr ready to send msgs")
        self.from_frame_rel = 'tag36h11:0'
        self.to_frame_rel = 'camera_link'
        self.timer  = self.create_timer(0.001,self.on_timer)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)
        self.l1 = 100
        self.l2 = 200
        self.x = 0
        self.y = 0
        self.z = 0
        
    def publsh_usr_inpt(self):

        x_val = self.x
        y_val = self.y
        z_val = self.z
        xy_val = math.sqrt(math.pow(x_val,2)+ math.pow(y_val,2))
        # belw is for obtaining theta2, theta1 and theta0
        num = xy_val * xy_val + z_val * z_val - self.l1 * self.l1 - self.l2 * self.l2
        den = 2*self.l1*self.l2
        theta2  = np.arccos(num/den) # dis is for link 2
        theta1 = np.arctan((self.l2*np.sin(theta2))/(self.l1 + self.l2 * np.cos(theta2))) # dis is for link 1

 
        angle_radians = math.atan2(y_val,x_val)
        num_two = angle_radians*180
        den_two = 3.14
        theta0 = num_two/den_two # dis is for waist to shoulder angle
        print(num_two/den_two)


        msg = String()

        msg.data = 'f' + ' ' + str(theta0) + str(theta1) + str(theta2) +'000000v' 
        self.publisher.publish(msg)
        self.get_logger().info(f'Publshn {msg.data}')
    
    def on_timer(self):
        try:
            #self.tf_buffer.can_transform(self.to_frame_rel, self.from_frame_rel, rclpy.time.Time(), timeout=1.0)
        
            t = self.tf_buffer.lookup_transform('robot_arm_bl','tag36h11:0',rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().info('could not transform')
            return
        self.x =  t.transform.translation.x
        self.y = -t.transform.translation.y
        self.z = t.transform.translation.z
        self.publsh_usr_inpt()

        #print(self.euler_from_quaternion(t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w))
        

    

def main(args=None):
    rclpy.init(args=args)
    nod = user_inpt_publshr()
    try:
        while True:
            rclpy.spin(nod) # please keep angle format as f 010020030040050v whre nos stands as angle values in dgrs , here 010, 020 stands for 10 and 20 dgrs of waist and shouldr rspctvly
    except KeyboardInterrupt:
        pass
    finally:
        nod.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()