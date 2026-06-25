## How do we design the hardware software interface for the UR Table.
 
It should be a distributed system. Implemented with ZMQ for speed and simplicity.

Robot Node
Publishes joint positions and their metadata (time of collection)
Subscribes to actions

Camera node
Publishes: top, wrist frames and their metadata (time of collection)
- Option to visualize the frames on the computer
 
Broker Node:
Aligns the metadata of the joint states and the camera.
It publishes the aligned metadata of the joint states and the camera

Environment
This is a gym environment that a policy like OGPO or openpi uses.
It subscribes to the broker topic for the aligned metadata and then draws the camera and joint position info of the timesteps that the aligned metadata from the broker said. 
It publishes actions
