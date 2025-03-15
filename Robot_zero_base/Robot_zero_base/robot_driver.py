#!/usr/bin/env python3
# This file isn't used as there are errors occuring when it runs but can be resolved after solving it if needed. Please use diff_tf.py which is an alternative for this file.
import serial
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
import rclpy
from rclpy import time
from rclpy.node import Node
from std_msgs.msg import Int16
import math
import numpy as np # Scientific computing library for Python
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class Encoder_publishr_subscrbr(Node):
    def __init__(self):
        super().__init__('encoder_publshr_subscrbr')
        self.ard = serial.Serial("/dev/ttyUSB0",115200)
        #self.publishr = self.create_publisher(Int16,'encoder_ticks',10) isn't used
        self.publshr_odom = self.create_publisher(Odometry,'odom',10)
        self.odom_broadcstr = TransformBroadcaster(self)
        self.get_logger().info('Encoder Publisher Node has been started.')
        self.timer = self.create_timer(0.1,self.publish_odomtry)
        self.pwm_motor_right_old = 0
        self.pwm_motor_left_old = 0
        self.ticks_per_revolutn_left_wheel = 367
        self.ticks_per_revolutn_right_wheel = 367
        # self.radius_left_wheel = 6.2 dis isn't used curntly
        # self.radius_right_wheel = 6.2
        self.distnc_lft = 0
        self.distnc_rght = 0
        self.disnc_trvld_by_cntr_pt_of_rbt = 0
        self.robot_turn_angle = 0
        self.wheel_base = 0.45 # cntr of left tire to cntr of rght tire
        self.odom_new = Odometry()
        self.odom_old = Odometry()
        self.old_robot_turn_angle = 0
        
        # new odometry data filling
        self.odom_new.header.frame_id = "odom"
        
        self.odom_new.pose.pose.position.z = 0.0
        self.odom_new.pose.pose.orientation.x = 0.0
        self.odom_new.pose.pose.orientation.y = 0.0
        
        self.odom_new.twist.twist.linear.x = 0.0
        self.odom_new.twist.twist.linear.y = 0.0
        self.odom_new.twist.twist.linear.z = 0.0
        
        self.odom_new.twist.twist.angular.x = 0.0
        self.odom_new.twist.twist.angular.y = 0.0
        self.odom_new.twist.twist.angular.z = 0.0

        # old odometry data filling
        self.odom_old.pose.pose.position.x = 0.0
        self.odom_old.pose.pose.position.y = 0.0
        self.odom_old.pose.pose.orientation.z = 0.0



    # def write_ser_helpr(self,cmd): useful for giving input to arduino but not useful curntly here.
    #     self.ard.write(cmd)

    def update_odom(self):
        # calcult avg distnc
        self.disnc_trvld_by_cntr_pt_of_rbt = (self.distnc_lft + self.distnc_rght)/2
        
        # calclt numbr of radians robot has turnd since last calcultn cycle
        # if (self.distnc_rght - self.distnc_lft) < -1 or (self.distnc_rght - self.distnc_lft) > 1: # values check if under bounds. Values mentioned here lead to out of bound values
        #     self.robot_turn_angle = self.old_robot_turn_angle
        # else: #put values
        self.robot_turn_angle = math.asin(math.radians((self.distnc_rght - self.distnc_lft)/self.wheel_base))
        #self.old_robot_turn_angle = self.robot_turn_angle

        if self.robot_turn_angle > 3.14:
            self.robot_turn_angle -=2*3.14
        elif self.robot_turn_angle < -3.14:
            self.robot_turn_angle +=2*3.14
        else:
            pass

        # calcult new pose (x,y and theta)
        self.odom_new.pose.pose.position.x = self.odom_old.pose.pose.position.x + math.cos(self.robot_turn_angle) * self.disnc_trvld_by_cntr_pt_of_rbt
        self.odom_new.pose.pose.position.y = self.odom_old.pose.pose.position.y + math.sin(self.robot_turn_angle) * self.disnc_trvld_by_cntr_pt_of_rbt
        self.odom_new.pose.pose.orientation.z = self.robot_turn_angle + self.odom_old.pose.pose.orientation.z

        # prevnt lockup due to a bad calcultn cycle
        if math.isnan(self.odom_new.pose.pose.position.x) or math.isnan(self.odom_new.pose.pose.position.y) or math.isnan(self.odom_new.pose.pose.position.z):
            self.odom_new.pose.pose.position.x = self.odom_old.pose.pose.position.x
            self.odom_new.pose.pose.position.y = self.odom_old.pose.pose.position.y
            self.odom_new.pose.pose.orientation.z = self.odom_old.pose.pose.orientation.z
        
        # make sure theta stays in correct range
        if self.odom_new.pose.pose.orientation.z > 3.14:
            self.odom_new.pose.pose.orientation.z -= 2 * 3.14
        elif self.odom_new.pose.pose.orientation.z < -3.14:
            self.odom_new.pose.pose.orientation.z += 2 * 3.14
        else:
            pass

        # compute velocity
        self.odom_new.header.stamp = self.get_clock().now().to_msg()
        self.odom_new.twist.twist.linear.x = self.disnc_trvld_by_cntr_pt_of_rbt/(self.odom_new.header.stamp.sec - self.odom_old.header.stamp.sec)
        self.odom_new.twist.twist.angular.z = self.disnc_trvld_by_cntr_pt_of_rbt / (self.odom_new.header.stamp.sec - self.odom_old.header.stamp.sec)

        # save pozn data for nxt calculatn cycl
        self.odom_old.pose.pose.position.x = self.odom_new.pose.pose.position.x
        self.odom_old.pose.pose.position.y = self.odom_new.pose.pose.position.y
        self.odom_old.pose.pose.orientation.z = self.odom_new.pose.pose.orientation.z
        
        
    def get_quaternion_from_euler(self,roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]

    def publish_odomtry(self):
        line = self.ard.readline().decode("utf-8").strip()
        temp  = line.split(',')
        print(temp)
        if temp[0] == '' or temp[1] == '':
            pwm_motor_right_fresh = self.pwm_motor_right_old
            pwm_motor_left_fresh = self.pwm_motor_left_old
        else:
            pwm_motor_right_fresh = int(temp[0]) # when I keep dis +
            pwm_motor_left_fresh = -int(temp[1]) # and dis -ve, I get both + pwm values when moved forward
        print('pwm_motor_right_fresh is')
        print(pwm_motor_right_fresh)
        print('pwm_motor_left_fresh is')
        print(pwm_motor_left_fresh)
        
        # pwm difrnc of left and rght well in curnt and past calcultn cycle of L and R motors
        pwm_motor_right_diff = pwm_motor_right_fresh - self.pwm_motor_right_old
        pwm_motor_left_diff = pwm_motor_left_fresh - self.pwm_motor_left_old
        
        # keep old (earlier frsh) pwm values in old designtd variables of L and R motors
        self.pwm_motor_right_old = pwm_motor_right_fresh
        self.pwm_motor_left_old = pwm_motor_left_fresh
        
        # find distnc trvld by L and R wheel 
        self.distnc_lft = pwm_motor_left_diff/ self.ticks_per_revolutn_left_wheel
        self.distnc_rght = pwm_motor_right_diff / self.ticks_per_revolutn_right_wheel
    
        self.update_odom()
        print("self.odom_new.pose.pose.orientation.z is ",self.odom_new.pose.pose.orientation.z)
        quatrnion_convrtd_angls = self.get_quaternion_from_euler(0,0,self.odom_new.pose.pose.orientation.z)
        quatrnon = Quaternion()
        quatrnon.x = 0.0
        quatrnon.y = 0.0
        quatrnon.z = quatrnion_convrtd_angls[2]
        quatrnon.w = quatrnion_convrtd_angls[3]

        self.odom_dta = Odometry()
        self.odom_dta.header.stamp = self.get_clock().now().to_msg()
        self.odom_dta.header.frame_id = 'odom'
        self.odom_dta.child_frame_id = 'base_link'
        self.odom_dta.pose.pose.position.x = self.odom_new.pose.pose.position.x
        self.odom_dta.pose.pose.position.y = self.odom_new.pose.pose.position.y
        self.odom_dta.pose.pose.position.z = 0.0
        self.odom_dta.pose.pose.orientation = quatrnon
        self.odom_dta.twist.twist.linear.x = self.odom_new.twist.twist.linear.x
        self.odom_dta.twist.twist.linear.y = 0.0
        self.odom_dta.twist.twist.angular.z = self.odom_new.twist.twist.angular.z
        self.publshr_odom.publish(self.odom_dta)

        trnsfm_stmpd_msg = TransformStamped()
        trnsfm_stmpd_msg.header.stamp = self.get_clock().now().to_msg()
        trnsfm_stmpd_msg.header.frame_id = 'base_link'
        trnsfm_stmpd_msg.child_frame_id = 'odom'
        trnsfm_stmpd_msg.transform.translation.x = self.odom_new.pose.pose.position.x
        trnsfm_stmpd_msg.transform.translation.y = self.odom_new.pose.pose.position.y
        trnsfm_stmpd_msg.transform.translation.z = 0.0
        trnsfm_stmpd_msg.transform.rotation.x = quatrnon.x
        trnsfm_stmpd_msg.transform.rotation.y = quatrnon.y
        trnsfm_stmpd_msg.transform.rotation.z = quatrnon.z
        trnsfm_stmpd_msg.transform.rotation.w = quatrnon.w

        self.odom_broadcstr.sendTransform(trnsfm_stmpd_msg)

    
def main(args = None):
    rclpy.init(args=args)
    node =  Encoder_publishr_subscrbr()
    try:
            rclpy.spin(node)
    except KeyboardInterrupt:
         pass
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
     main()





