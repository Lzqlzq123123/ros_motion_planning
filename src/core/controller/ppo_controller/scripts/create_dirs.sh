#!/bin/bash
# Create necessary directories for PPO controller

PACKAGE_PATH=$(rospack find ppo_controller)

# Create models directory
mkdir -p ${PACKAGE_PATH}/models

# Create logs directory
mkdir -p ${PACKAGE_PATH}/logs/tensorboard

# Create results directory
mkdir -p ${PACKAGE_PATH}/results

echo "Directories created successfully!"
echo "Models: ${PACKAGE_PATH}/models"
echo "Logs: ${PACKAGE_PATH}/logs"
echo "Results: ${PACKAGE_PATH}/results"