#!/usr/bin/env python


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped
import serial
from math import sin, cos
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np
import math
NS_TO_SEC= 1000000000

class TwistToMotors(Node):
    """
    twist_to_motors - converts a twist message to motor commands.  Needed for navigation stack

    """

    def __init__(self):
        super(TwistToMotors, self).__init__("twist_to_motors")
        self.nodename = "twist_to_motors"
        self.get_logger().info("%s started" % self.nodename)
        self.ard_write = serial.Serial("/dev/ttyUSB0",115200) #pls chng dis as pr ESP32 usng | USB0 ESP32 stnds for motor driver ESP32
        self.ard_read_two = serial.Serial("/dev/ttyUSB1",115200) # USB1 ESP32 stands for encoder ESP32
        self.dx = 0.0
        self.dr = 0.0
        self.velct_min = 0.05
        self.velct_max = 0.5
        self.pwm_min = 40
        self.pwm_max = 255
        self.countr = 0
        self.pwm = 0
        self.encoder_ticks_left_fresh = 0
        self.encoder_ticks_right_fresh = 0
        self.d_lft = 0
        self.d_rght = 0
        self.d = 0


        #### parameters #######
        self.rate_hz = self.declare_parameter("rate_hz", 10.0).value # the rate at which to publish the transform


        self.ticks_meter = float(self.declare_parameter('ticks_meter', 933).value)  # The number of wheel encoder ticks per meter of travel
        self.base_width = float(self.declare_parameter('base_width', 0.45).value)  # The wheel base width in meters

        self.base_frame_id = self.declare_parameter('base_frame_id','base_link').value  # the name of the base frame of the robot
        self.odom_frame_id = self.declare_parameter('odom_frame_id','odom').value  # the name of the odometry reference frame

        self.encoder_min = self.declare_parameter('encoder_min', -32768).value
        self.encoder_max = self.declare_parameter('encoder_max', 32768).value
        self.encoder_low_wrap = self.declare_parameter('wheel_low_wrap', (self.encoder_max - self.encoder_min) * 0.3 + self.encoder_min).value
        self.encoder_high_wrap = self.declare_parameter('wheel_high_wrap', (self.encoder_max - self.encoder_min) * 0.7 + self.encoder_min).value

        self.encoder_ticks_right_old = 0
        self.encoder_ticks_left_old = 0

        # internal data
        self.enc_left = None  # wheel encoder readings
        self.enc_right = None
        self.left = 0.0  # actual values coming back from robot
        self.right = 0.0
        self.lmult = 0.0
        self.rmult = 0.0
        self.prevous_lft_encodr = 0
        self.prevous_rght_encodr = 0
        self.x = 0.0  # position in xy plane
        self.y = 0.0
        self.th = 0.0
        self.dx = 0.0  # speeds in x/rotation
        self.dr = 0.0
        self.then = self.get_clock().now()

        self.left_and_rght_data_to_strng = 'c0,0'

        self.drive_cmd_vel = [0] *2

        self.rbt_arm_strn = 'f 000000000000000'

        # publisher and timer
        self.create_timer(0.002,self.fetch_encoder_data)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.odom_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Twist, '/cmd_vel', self.twist_callback, 10)
        self.create_subscription(String,'robot_arm_angles_here',self.give_robt_arm_angles,10)
        self.create_timer(0.02, self.update)


    def velct_to_pwm(self,vel):

        self.pwm = self.pwm_min + ((vel - self.velct_min)/(self.velct_max - self.velct_min)) * (self.pwm_max - self.pwm_min)

        return self.pwm

    def quaternion_from_euler(self,roll, pitch, yaw):
        """
        Converts euler roll, pitch, yaw to quaternion (w in last place)
        quat = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr

        return q

    def update(self):


        now = self.get_clock().now()
        elapsed = now - self.then
        self.then = now
        elapsed = elapsed.nanoseconds / NS_TO_SEC

        # calculate odometry
        if self.enc_left == None:
            d_left = 0
            d_right = 0
        else:
            d_left = (self.left - self.enc_left) / self.ticks_meter  # dis value is zero, dus it seems useless
            d_right = (self.right - self.enc_right) / self.ticks_meter # dis value is zero, dus it seems useless
        self.d_lft = d_left
        self.d_rght = d_right
        self.enc_left = self.left
        self.enc_right = self.right

        # distance traveled is the average of the two wheels 
        d = (d_left + d_right) / 2
        self.d = d # 'd' value is zero
        # this approximation works (in radians) for small angles
        th = (d_right - d_left) / self.base_width
        # calculate velocities
        self.dx = d / elapsed
        self.dr = th / elapsed

        if d != 0: # 0 !=0 means dis condtn is true
            # calculate distance traveled in x and y
            x = cos(th) * d
            y = -sin(th) * d
            # calculate the final position of the robot
            self.x = self.x + (cos(self.th) * x - sin(self.th) * y)
            self.y = self.y + (sin(self.th) * x + cos(self.th) * y)
        if th != 0:
            self.th = self.th + th
        print('self.th is ',self.th)



        # publish the odom information
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = sin(self.th / 2)
        quaternion.w = cos(self.th / 2)
        # [r,p,yaw] = self.euler_from_quaternion(quaternion.x ,quaternion.y ,quaternion.z ,quaternion.w) using dis only for my conveninc

        # [angle_one,angle_two,angle_three,angle_four] = self.quaternion_from_euler(r,p,yaw)

        transform_stamped_msg = TransformStamped()
        transform_stamped_msg.header.stamp = self.get_clock().now().to_msg()
        transform_stamped_msg.header.frame_id = self.odom_frame_id 
        transform_stamped_msg.child_frame_id = self.base_frame_id
        transform_stamped_msg.transform.translation.x = self.x
        transform_stamped_msg.transform.translation.y = self.y
        transform_stamped_msg.transform.translation.z = 0.0625  #wheel diamtr / 2
        transform_stamped_msg.transform.rotation.x = quaternion.x
        transform_stamped_msg.transform.rotation.y = quaternion.y
        transform_stamped_msg.transform.rotation.z = quaternion.z
        transform_stamped_msg.transform.rotation.w = quaternion.w

        self.odom_broadcaster.sendTransform(transform_stamped_msg)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = quaternion
        odom.child_frame_id = self.base_frame_id
        odom.twist.twist.linear.x = self.dx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = self.dr
        self.odom_pub.publish(odom)


    def euler_from_quaternion(self,x, y, z, w): # dis isn't used
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def fetch_encoder_data(self):
        line = self.ard_read_two.readline().decode("utf-8",errors='ignore').strip()
        print('self.d_lft is')
        print(self.d_lft)
        print('self.d_right is')
        print(self.d_rght)
        print('self.d is ')
        print(self.d)
        print('line is ')
        print(line)
        try:
            if len(line) != 0:
                if line[0] == 'e':
                    gh = line.replace('e','')
                    temp  = gh.split(',')
                    print('temp is below')
                    print(temp)
                if len(temp) == 2:
                    if temp[0] == '' or temp[1] == '':
                        self.encoder_ticks_right_fresh = self.encoder_ticks_right_old
                        self.encoder_ticks_left_fresh = self.encoder_ticks_left_old
                    else:
                        self.encoder_ticks_right_fresh = int(temp[1]) # when I keep dis +
                        self.encoder_ticks_left_fresh = -int(temp[0]) # and dis -ve, I get both + pwm values when moved forward
            else:
                self.encoder_ticks_left_fresh = self.encoder_ticks_left_old
                self.encoder_ticks_right_fresh = self.encoder_ticks_right_old
        except NameError as ne:
            print('error in fetching data, dus ignoring' ,ne)
            self.encoder_ticks_right_fresh = self.encoder_ticks_right_old
            self.encoder_ticks_left_fresh = self.encoder_ticks_left_old
        except ValueError as ve:
            print('ignoring ',ve)
            self.encoder_ticks_right_fresh = self.encoder_ticks_right_old
            self.encoder_ticks_left_fresh = self.encoder_ticks_left_old
        
        self.encoder_ticks_right_old = self.encoder_ticks_right_fresh
        self.encoder_ticks_left_old = self.encoder_ticks_left_fresh
        print('encoder ticks right is')
        print(self.encoder_ticks_right_fresh)
        print('encoder ticks left is')
        print(self.encoder_ticks_left_fresh)
        enc_lft = self.encoder_ticks_left_fresh
        if enc_lft < self.encoder_low_wrap and self.prevous_lft_encodr > self.encoder_high_wrap:
            self.lmult = self.lmult + 1

        if enc_lft > self.encoder_high_wrap and self.prevous_lft_encodr < self.encoder_low_wrap:
            self.lmult = self.lmult - 1

        self.left = 1.0 * (enc_lft + self.lmult * (self.encoder_max - self.encoder_min))
        self.prevous_lft_encodr = enc_lft

        enc_rght = self.encoder_ticks_right_fresh
        if enc_rght < self.encoder_low_wrap and self.prevous_rght_encodr > self.encoder_high_wrap:
            self.rmult = self.rmult + 1

        if enc_rght > self.encoder_high_wrap and self.prevous_rght_encodr < self.encoder_low_wrap:
            self.rmult = self.rmult - 1

        self.right = 1.0 * (enc_rght + self.rmult * (self.encoder_max - self.encoder_min))
        self.prevous_rght_encodr = enc_rght

        # print('self.countr is ',self.countr)
        # self.countr = self.countr + 1
        # if self.countr == 700:
        #     left_and_rght_data_to_strng = 'c80,80'
        #     self.ard.write(left_and_rght_data_to_strng.encode())
        # if self.countr == 1500:
        #     left_and_rght_data_to_strng = 'c120,120'
        #     self.ard.write(left_and_rght_data_to_strng.encode())
        # if self.countr == 2000:
        #     left_and_rght_data_to_strng = 'c0,0'
        #     self.ard.write(left_and_rght_data_to_strng.encode())
        #     self.countr = 0

        # Send motor pwm signal


        # #if self.dx == 0.0 and self.dr == 0.0:
        # if self.countr < 800:
        #     left_and_rght_data_to_strng = 'c0,0'
        #     self.ard.write(left_and_rght_data_to_strng.encode())
        # else:
        #     self.calculate_left_and_right_target_and_give_values()
    

    def calculate_left_and_right_target_and_give_values(self):
        # dx = (l + r) / 2
        # dr = (r - l) / w

        right = Float32()
        left = Float32()
        
        right.data = self.dx # dis is useless I thnk since right.data is modfd

        right.data = 1.0 * self.dx - self.dr * self.base_width / 2.0 # sign change at self.dx - self.dr is crucial as while turning clk or ant clk of robot, dese values change
        left.data = 1.0 * self.dx + self.dr * self.base_width / 2.0  # sign change at self.dx + self.dr is crucial as while turning clk or ant clk of robot, dese values change

        print('right.data vel is ',right.data)
        print('left.data vel is ',left.data)
        intrplt_right_data = int(self.velct_to_pwm(right.data))
        intrplt_left_data = int(self.velct_to_pwm(left.data))
        print('intrplt_right_data (which is pwm for right whel) is ',intrplt_right_data)
        print('intrplt_left_data (which is pwm for left whel) is ',intrplt_left_data)
        if intrplt_left_data > 255:
            intrplt_left_data = 255
        if intrplt_left_data < -255:
            intrplt_left_data = -255
        if intrplt_right_data > 255:
            intrplt_right_data = 255
        if intrplt_right_data < -255:
            intrplt_right_data = -255


        right_data_to_Strng = str(intrplt_right_data)
        left_data_to_Strng = str(intrplt_left_data)

        self.left_and_rght_data_to_strng = 'o'+ ' ' + left_data_to_Strng + ' ' + right_data_to_Strng + 'v'
        print('left_and_rght_data_to_strng is ',self.left_and_rght_data_to_strng)
        self.send_dta()
    
    def send_dta(self):
        self.ard_write.write(self.left_and_rght_data_to_strng.encode())
        self.get_logger().error("SENT SERIAL PWM DATA")

    def give_robt_arm_angles(self,msg):
        if len(msg.data) > 0:
            self.rbt_arm_strn = msg.data
            self.ard_write.write(self.rbt_arm_strn.encode())

        else:
            self.rbt_arm_strn = 'f 000000000000000'


    def twist_callback(self, msg):
        # self.countr = self.countr + 1
        # if self.countr == 10000:
        #     self.countr = 0
        print('self.dx is ')
        print(self.dx)
        print('self.dr is ')
        print(self.dr)
        self.dx = msg.linear.x
        self.dr = msg.angular.z
        if self.dx == 0.0 and self.dr == 0.0:
        # if self.countr < 800:
            left_and_rght_data_to_strng = 'o 0 0v'
            self.ard_write.write(left_and_rght_data_to_strng.encode())
        else:
            self.calculate_left_and_right_target_and_give_values()

        # if self.countr == 500:
        #     left_and_rght_data_to_strng = 'c100,100'
        #     self.ard.write(left_and_rght_data_to_strng.encode())





def main(args=None):
    rclpy.init(args=args)
    try:
        twist_to_motors = TwistToMotors()
       # twist_to_motors.create_rate(50)
        rclpy.spin(twist_to_motors)
    except rclpy.exceptions.ROSInterruptException:
        pass

    twist_to_motors.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()