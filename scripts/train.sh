#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
TASK=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

python train.py task=AllegroHandHora headless=True seed=${SEED} \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
task.env.object.type=cylinder_default \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
device_id="cuda:${GPUS}" \
graphics_device_id=${GPUS} \
train="${TASK}" \
${EXTRA_ARGS}