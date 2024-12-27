from threading import Thread, Event
import numpy as np
import time
from pynput import keyboard
import rclpy
from rclpy.node import Node

class ActionPublisher(Node):
    def __init__(self):
        super().__init__('robot_command_publisher')
        self.get_logger().info('robot interface')
        # self.publish=self.create_publisher()
    def get_keyborad(self, key):
        """Update the current pose and publish it."""
        if(key=='a'):
            print('rotate left')
        elif(key=='d'):
            print('rotate right')
        elif(key=='w'):
            print('rotate up')
        elif(key=='s'):
            print('rotate down')
        else:    
            print("no define "+key)

class KeyboardInput(Thread):
    def __init__(self, ros2_interface):
        super().__init__()
        self.stop_event = Event()
        self.active_keys = set()
        self.publish=ros2_interface
    def stop(self):
        """Stop the thread."""
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def on_press(self, key):
        """Handle key press events."""
        try:
            pass
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            key_name = key.char if hasattr(key, 'char') else key.name
            # self.active_keys.discard(key_name)
            self.publish.get_keyborad(key_name)
        except AttributeError:
            pass

    def run(self):
        """Run the thread."""
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while not self.stop_event.is_set():
                time.sleep(0.1)  # Polling frequency
            listener.stop()


def main():
    rclpy.init()
    action_publisher = ActionPublisher()
    with KeyboardInput(action_publisher) as ki:
        try:
            rclpy.spin(action_publisher)
        except KeyboardInterrupt:
            pass



if __name__ == '__main__':
    main()