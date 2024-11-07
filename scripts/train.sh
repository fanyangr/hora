#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python train.py task=AllegroHandHora headless=True seed=0 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
task.env.object.type=cylinder_default \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=AllegroHandHora/test \
${EXTRA_ARGS}