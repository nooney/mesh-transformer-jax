#!/bin/bash

# Args:
# 0: ./tamsin_script.sh
# 1: project-name
# 2: zone 
# 3: tpu-name
# 4: accelerator-type 
# 5: version
# 6: docker container name


# Create GC storage
# Create a Project
# gcloud config set project "$1"
# gcloud config set compute/zone "$2"


# Check if TPU exists
description=$(gcloud compute tpus tpu-vm describe "$3" --zone="$2")
sub="state: "
state=${description#*$sub}
if [ "$state" == "READY" ]
then
    echo "TPU exists and is READY"
elif [ "$state" == "STOPPED" ]
then 
    echo "TPU exists and is STOPPED. Starting TPU $3"
    gcloud compute tpus tpu-vm start "$3"
else
    # Create TPU
    echo "TPU does not exist. Creating TPU $3"
    gcloud compute tpus tpu-vm create "$3" --zone="$2" --accelerator-type="$4" --version="$5"
fi

# SSH into TPU VM
echo "SSH into $3"
gcloud compute tpus tpu-vm ssh "$3" --zone="$2"

#Start Docker daemon in TPU VM
sudo systemctl start docker

# Start Docker container
sudo docker run -ti --rm --name $6 --privileged --network=host python:3.8 bash

# Set up
# echo  "Check startup script"
# sh ./tamsin_setup.sh

# # Run training
# python3  device_train.py --config=configs/pmqs_config.json --tune-model-path=gs://nlp-gpt-j-bucket/step_383500/

# # Save results to storage
# # Stop TPU
# echo "Stopping TPU: $3"
# gcloud compute tpus tpu-vm stop "$3" --zone="$2"
# # Delete TPU
# echo "Deleting TPU: $3"
# gcloud compute tpus tpu-vm delete "$3" --zone="$2"