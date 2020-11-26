## Steps to run the RoboNav

- Set the TURTLEBOT3_MODEL in system environment\
`export TURTLEBOT3_MODEL=burger`

- Launch Gazebo with turtlebot3 in stage 1 environment\
`roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch`

- Launch the DQN training\
`roslaunch turtlebot3_dqn turtlebot3_dqn_stage_1_torch.launch`
