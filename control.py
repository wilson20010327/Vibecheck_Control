
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
import csv
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from threading import Thread, Event
import time
from classifior import Classifior

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
        self.recording_interval = 1  # seconds interval between publishing messages
        self.max_publish_count = 1  # Number of times to publish messages
        self.message_count = 0  # Initialize message count
        self.current_publish_count = 0  # Count of publish commands
        self.df=[]
        self.record=False
    def publish_command(self):

        # clean cache
        self.df=[]
        
        # define start msg
        msg = Bool()
        msg.data = True
        # trigger to teesny
        self.record=True
        self.publisher.publish(msg)
        time.sleep(1)
        self.record=False
            
    def listener_callback(self, msg):
        if(not self.record):
            return
        normalized_data = [(x * 1) for x in msg.data]
        for i in normalized_data:
            self.df.append(i)
        
    def get_df(self):
        return self.df

def main(args=None):
    rclpy.init(args=args)
    node = CommandAndDataCollector()
    
    model_folder = "/root/ur5/microros_ws/model/"
    classifier = Classifior(model_path=model_folder)
    classifier.load_model("mlp_insertion2_acc_90", "kpca_insertion2_acc_90")

    with Test(node, classifier) as t:
        try:
            rclpy.spin(node)  # Use spin_once to allow shutdown to break the loop
        except:
            rclpy.shutdown()





class Test(Thread):
    def __init__(self, ros2_interface, classifier):
        super().__init__()
        self.publish=ros2_interface
        self.classifier = classifier
    
    def run(self):
        """Run the thread."""
        time.sleep(0.5)
        self.publish.publish_command()
        data = self.publish.get_df()
        print(data,len(data))
        
        # Use Classifior to predict
        self.classifier.set_data(data)
        self.classifier.processData()
        label = self.classifier.predict_label()
        
        
    def stop(self):
        """Stop the thread."""
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
   
        

if __name__ == '__main__':
    main()