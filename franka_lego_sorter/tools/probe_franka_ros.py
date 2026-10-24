#!/usr/bin/env python3
"""Read-only ROS discovery/state probe: no action goals, launches, or writes to robot."""
import json
import time
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers
from tf2_ros import Buffer, TransformListener
from rosidl_runtime_py.convert import message_to_ordereddict


def main():
    rclpy.init()
    node = rclpy.create_node('lego_commissioning_read_only_probe')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    joint_messages = {}
    subscriptions = []
    clients, futures = {}, {}
    started = time.monotonic()
    deadline = started + 10.
    seen = set()
    try:
        while time.monotonic() < deadline:
            topics = node.get_topic_names_and_types()
            for name, types in topics:
                if name not in seen and 'sensor_msgs/msg/JointState' in types:
                    seen.add(name)
                    def callback(message, topic=name):
                        joint_messages[topic] = message_to_ordereddict(message)
                    subscriptions.append(node.create_subscription(
                        JointState, name, callback, qos_profile_sensor_data))
            for name, types in node.get_service_names_and_types():
                if name not in clients and 'controller_manager_msgs/srv/ListControllers' in types:
                    clients[name] = node.create_client(ListControllers, name)
                if name in clients and name not in futures and clients[name].service_is_ready():
                    futures[name] = clients[name].call_async(ListControllers.Request())
            rclpy.spin_once(node, timeout_sec=.1)
        report = dict(nodes=node.get_node_names_and_namespaces(),
                      topics=node.get_topic_names_and_types(),
                      services=node.get_service_names_and_types(),
                      joint_states=joint_messages, controllers={})
        for name, future in futures.items():
            if future.done():
                try:
                    report['controllers'][name] = message_to_ordereddict(future.result())
                except Exception as error:
                    report['controllers'][name] = {'error': str(error)}
        report['transforms'] = {}
        for parent, child in [('fr3_link0', 'fr3_hand_tcp'), ('fr3_link8', 'fr3_hand_tcp'),
                              ('fr3_link0', 'fr3_link8'), ('fr3_hand', 'fr3_hand_tcp')]:
            try:
                tf = buffer.lookup_transform(parent, child, Time())
                report['transforms'][parent + '__' + child] = message_to_ordereddict(tf)
            except Exception as error:
                report['transforms'][parent + '__' + child] = {'error': str(error)}
        path = Path(__file__).resolve().parents[1] / 'data/commissioning/ros-host-state.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        listener.unregister()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
