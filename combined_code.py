######################################

# !/usr/bin/python3

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
from classifior import Classifier
class CommandAndDataCollector(Node):
    def __init__(self):
        super().__init__('command_and_data_collector')

        # QoS profile for data subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Create a publisher to publish the Bool message (control the sweep)
        self.publisher = self.create_publisher(Bool, 'command_topic', 10)

        # Create a subscription to subscribe to the Int16MultiArray messages
        self.subscription = self.create_subscription(
            Int16MultiArray,
            '/micro_ros_arduino_node_publisher',
            self.listener_callback,
            qos_profile=qos_profile
        )

        # CSV file setup
        self.csv_file_prefix = 'output'
        self.recording_interval = 1  # seconds interval between publishing messages
        self.max_publish_count = 5  # Number of times to publish messages
        self.message_count = 0  # Initialize message count
        self.samples_per_msg = 128  # Number of samples per message
        self.current_publish_count = 0  # Count of publish commands

        # Time settings
        self.start_time = None  # Initialize start time as None

        # Timer to publish messages
        # self.publish_timer = self.create_timer(self.recording_interval, self.publish_command)
        
    def start_timer(self):
        self.publish_timer = self.create_timer(self.recording_interval, self.publish_command)

    def get_unique_csv_file_name(self):
        # Get all files in the current directory
        existing_files = os.listdir()
        # Find all files that match the pattern "output_#.csv"
        matching_files = [f for f in existing_files if f.startswith(self.csv_file_prefix) and f.endswith('.csv')]

        # If there are no matching files, start with "output_1.csv"
        if not matching_files:
            return f'{self.csv_file_prefix}_1.csv'

        # Extract the numbersrclpy from the file names
        existing_numbers = []
        for file_name in matching_files:
            try:
                # Get the number between "output_" and ".csv"
                number = int(file_name.split('_')[1].split('.')[0])
                existing_numbers.append(number)
            except (ValueError, IndexError):
                continue

        # Determine the next available number
        next_number = max(existing_numbers, default=0) + 1
        return f'{self.csv_file_prefix}_{next_number}.csv'

    def init_csv_file(self):
        # Create a new CSV file for each publish with a unique name
        csv_file_path = self.get_unique_csv_file_name()
        self.current_csv_file_path = csv_file_path  # Save the current file path for use in listener_callback
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['msg', 'time(second)'] + [f'data_{i}' for i in range(self.samples_per_msg)])  # Set data size, the number of columns

    def publish_command(self):
        if self.current_publish_count < self.max_publish_count:
            self.current_publish_count += 1
            self.message_count = 0  # Reset message count for each publish
            self.start_time = self.get_clock().now()  # Record the start time for each publish

            # Initialize CSV file for this publish
            self.init_csv_file()

            # Publish command message
            msg = Bool()
            msg.data = True
            self.publisher.publish(msg)
            self.get_logger().info('Published command message #%d: %s' % (self.current_publish_count, msg.data))
        else:
            # Stop the timer after publishing the required number of times
            self.publish_timer.cancel()
            self.get_logger().info('Completed publishing %d messages.' % self.max_publish_count)
            self.destroy_node()  # Cleanly destroy the node
            raise Exception("Closed")

    def listener_callback(self, msg):
        if self.start_time is None:
            self.get_logger().warn('Start time is not set. Ignoring message.')
            return

        self.message_count += 1  # Message counter
        current_time = self.get_clock().now()
        elapsed_time = current_time - self.start_time  # Get elapsed time from the start
        elapsed_seconds = elapsed_time.nanoseconds / 1e9  # Convert nanoseconds to seconds

        # Map the incoming data from int16 values [-32768, 32767] to float values (if necessary)
        normalized_data = [(x * 1) for x in msg.data]

        # Write data into CSV file
        with open(self.current_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.message_count, elapsed_seconds] + normalized_data)

class EEPrimitives(Node):

  def __init__(self,exp, threshold_z=15, threshold_xy=13):
    super().__init__('ee_primitives_node')
    self.robot_ip = '192.168.0.216' # fixed

    # Initialize CommandAndDataCollector object
    self.data_collector = CommandAndDataCollector()

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
    # self.timer=self.create_timer(0.001, self.continuous_move_down)
    
  def contact_callback(self, msg):
        """Callback function for the contact detection topic."""
        self.get_logger().info(f"Contact detected: {msg .data}")
        if msg.data:
            self.contact_detected = True
            self.rtde_c.stopL()  # Stop linear motion immediately
            print("you should stop")
            # time.sleep(0.5)
            self.run_data_collection()

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

  def movement_based_on_label(self,label,exp):
    if label == 'diagonal_2points':
      self.rotation_z_axis(exp)

    elif label == 'one_line':
      self.rotation_x_axis(exp)

    elif label == 'one_surface':
        task.open_gripper()
        self.task_done = True
        print('Task completed')
    
    return exp

  
  def disconnect(self):
    self.get_logger().info('Disconnecting from robot.')
    self.rtde_c.stopScript()
  
  def run_data_collection(self):
    # Start the data collection process
    # self.data_collector.publish_command()
    self.data_collector.start_timer()
    self.get_logger().info('Timer is created!')
    rclpy.spin(self.data_collector)


def main(args=None):
    exp = [-0.5944156254820065, -0.09571589518438497, -0.08188298887537103, -122, 1, 35] 

    rclpy.init(args=args)
    # exp = [-0.575, -0.301, 0.2, -127, 0, 0] # initial position
    data_dir = '/root/ur5/microros_ws/data/test/'
    model_folder = "/root/ur5/microros_ws/model/"

    EE = EEPrimitives(exp)
    classifier = Classifier(data_path=data_dir, model_path=model_folder)
    classifier.load_model("mlp_insertion2_acc_90", "kpca_insertion2_acc_90")

    pose = EE.euler_to_rotvec_pose(exp)
    EE.move_l_to_start(pose)

    while not EE.task_done:
        try:
            time.sleep(2)
            exp = EE.continuous_move_down(exp)
            # implement a publisher function with while loop
            # t0 = threading.Thread(EE.publish_info)
            # t0.start()
            # loop:
            # while True:
            #     action = action_source
            #     ros.execute(action)
            #     colided = ros.get_colision()
            #     if collided:
            #          touch_data = ros.get_touch()
            #          c_type = classifier()
            #          action = get_action(c_type)
            rclpy.spin(EE)
        except:
            classifier.predict(data)
            label = classifier.predict_label()
            
            exp = EE.continuous_move_up(exp)
            exp = EE.movement_based_on_label(label,exp)
            

    EE.disconnect()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

