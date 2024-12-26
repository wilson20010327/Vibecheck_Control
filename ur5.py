import rtde_control # real time communication with robot
import rtde_receive
import numpy as np
import transforms3d as tf
from scipy.spatial.transform import Rotation as R
import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int16MultiArray
import csv
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy
import pandas as pd
import pickle
import re
class UR5_Action(Node):

  def __init__(self,exp, threshold_z=15, threshold_xy=13):
    super().__init__('ee_primitives_node')
    self.robot_ip = '192.168.0.216' # fixed

    ## set TCP offset
    # self.rtde_c.setTcp([0, 0, 0.0526, 0, 0, 0])
    self.step = -0.005
    self.pose = exp
    ## speed in joint and tool space are different (speed is slower in joint space)
    self.speedJ = 0.5  # m/s
    self.accelJ = 0.3  # m/s^2
    self.speedL = 0.1  # m/s
    self.accelL = 0.2  # m/s^2

    self.speedD = 0.05  # m/s
    self.accelD = 0.05  # m/s^2

    self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
    self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
    
    # Set TCP offset
    self.tcp_offset = [-0.023, 0, 0.25, 0, 0, 0]  # good one, move the EE frame to here
    self.rtde_c.setTcp(self.tcp_offset)
    self.tcp_pose = self.rtde_r.getActualTCPPose()

    self.threshhold_z = 0.05

    self.task_done = False

    # ROS2 subscriber
    self.contact_detected = False
    self.subscription = self.create_subscription(
        Bool,
        'contact_detected',
        self.contact_callback,
        10
    )
    
  def contact_callback(self, msg):
        """Callback function for the contact detection topic."""
        self.get_logger().info(f"Contact detected: {msg .data}")
        if msg.data:
            self.contact_detected = True
            self.rtde_c.stopL()  # Stop linear motion immediately
            print("you should stop")
            # time.sleep(0.5)
            self.trigger_data_collection()

  def euler_to_rotvec_pose(self, pose):

    axes, angle = tf.euler.euler2axangle(math.radians(pose[3]), math.radians(pose[4]), math.radians(pose[5]))
    rotvec = axes * angle
    pose[3:] = rotvec

    return pose

  def move_j_to_start(self, pose):
      """ Move to start joint pose """
      print('Moving to initial joint pose: ', pose)
      self.rtde_c.moveJ(pose, self.speedJ, self.accelJ)
      print('At start pose')


  def move_l_to_start(self, pose):
    """ move to start pose for experiments  """
    print('moving to start pose: ', pose)
    self.rtde_c.moveL(pose, self.speedL, self.accelL)
    print('at the pose for experiment!')

  def continuous_move_down(self,exp):
    """Move down incrementally along the z-axis."""

    exp[2] -= 0.032    # Set the limit to protect the grippers

    self.rtde_c.moveL(exp, self.speedD, self.accelD, asynchronous= True)
    # time.sleep(0.1)  # Small delay for smooth movement
    return exp

  def continuous_move_up(self,exp):
    """Move down incrementally along the z-axis."""

    exp[2] = -0.08188298887537103    # Set the limit to protect the grippers

    self.rtde_c.moveL(exp, self.speedL, self.accelL, asynchronous= True)

    return exp

  def rotation_z_axis(self,exp):

    exp[5] += 17.666666666666666667    # Set the limit to protect the grippers

    self.rtde_c.moveL(exp, self.speedD, self.accelD, asynchronous= True) 

    return exp  

  def rotation_x_axis(self,exp):

    exp[3] -= 19.333333333333333333

    self.rtde_c.moveL(exp, self.speedD, self.accelD, asynchronous= True) 

    return exp
    
  def trigger_data_collection(self):
    self.get_logger().info('Data collection start.')
    pass
  
  def disconnect(self):
    self.get_logger().info('Disconnecting from robot.')
    self.rtde_c.stopScript()
    
    
  def move_until_contact(self):
    speed = [0, 0, -0.100, 0, 0, 0] # self defined speed
    self.rtde_c.moveUntilContact(speed)
    print("Contact detected")
    
  def get_current_pose(self):
    return self.rtde_r.getActualTCPPose()
