# Global Development Log

## System Analysis (Initial)
- **Current State**: The project is a ROS-based Reinforcement Learning environment using `ros_gazebo_env.py` and `rsl_rl` for training a forklift agent.
- **Issue**: The goal marker is not published/visible in the first episode of training.
- **Hypothesis**: This is likely due to a race condition where the ROS publisher sends the message before the subscriber (RViz) is connected, or the `reset()` logic has a flaw handling the initial state. The `RobotAgent` initialization creates the publisher, and `reset()` is called shortly after. ROS publishers are asynchronous and may drop messages if sent immediately after creation if no subscribers are known yet (or latching is not used/configured). Although `queue_size=1` is set, without latching, late subscribers miss it. However, if the subscriber is already up (RViz open), the connection delay is the main suspect.

## TASK-212
- **Changes**: src/rl_training/envs/ros_gazebo_env.py: 261 -> Added `latch=True` to `goal_marker_pub`.
- **Line Stats**: +1, -1
- **Errors**: None
- **Context**: Solves the issue where the first goal marker is missing in RViz by latching the message so new subscribers receive the last published goal.

## System Analysis (Training Convergence)
- **Issue**: User reports training is not converging.
- **Analysis**:
    1.  **Reward Function**: The `collision_penalty` (-500.0) is extremely high compared to other rewards. This likely causes the agent to adopt a "safety-first" policy where it refuses to move to avoid any risk of collision, especially early in training when exploration is random. The `step_cost` (-0.5) is also significant.
    2.  **Observation**: `self.scan` is initialized to 100.0, but clamped to 10.0 in the callback. If the callback is late, the first observation contains values 10x larger than normal (10.0 vs 1.0 after normalization). This creates Out-Of-Distribution (OOD) noise.
    3.  **Model Complexity**: The current `[128, 128]` MLP might be too simple for a 367-dimensional input space, limiting the policy's expressiveness.
- **Plan**:
    1.  **Fix Observation**: Initialize `self.scan` to 10.0 to match the sensor's logical max range.
    2.  **Tune Rewards**: Reduce `collision_penalty` significantly (e.g., to -50 or -100) to encourage exploration.
    3.  **Upgrade Model**: Increase actor/critic network dimensions (e.g., `[512, 256, 128]`).

## TASK-213
- **Changes**: 
    - src/rl_training/config/forklift_ppo.yaml: 32 -> Reduced `collision_penalty` from -500.0 to -50.0; 73-74 -> Increased network dims to `[512, 256, 128]`.
    - src/rl_training/envs/ros_gazebo_env.py: 265 -> Changed `self.scan` init to 10.0.
- **Line Stats**: +4, -4
- **Errors**: None
- **Context**: Optimized training parameters. Reduced excessive punishment to allow exploration and increased model capacity for better generalization. Fixed scan initialization to be consistent with normalized observation range.

## TASK-214
- **Changes**: src/rl_training/config/forklift_ppo.yaml: 70-80 -> Changed policy class to `ActorCriticRecurrent` and added LSTM params.
- **Line Stats**: +5, -2
- **Errors**: None
- **Context**: Upgraded policy to use LSTM (Recurrent PPO). This allows the agent to maintain a memory of past observations (state), which is crucial for handling partial observability (e.g., dynamic obstacles, sensor noise, or velocity estimation).
