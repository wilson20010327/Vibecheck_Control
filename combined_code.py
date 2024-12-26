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

