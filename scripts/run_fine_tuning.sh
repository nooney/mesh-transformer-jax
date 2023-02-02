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

set -x
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
gcloud compute tpus tpu-vm ssh "$3" --zone="$2" --worker=all --command "
pip install -U pip && 
pip install -U jax[tpu]==0.2.18 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
pip install --no-dependencies optax==0.0.9 && 
pip install --no-dependencies chex==0.0.4 &&
pip install ray[default]==1.4.1 &&
pip install --no-dependencies dm-tree &&
pip install --no-dependencies toolz &&
pip install wandb==0.11.2 &&
pip install tqdm &&
pip install smart_open[gcs] &&
pip install typing-extensions==3.7.4.3 &&
pip install pydantic==1.8 &&
pip install protobuf~=3.20 &&
pip install google-api-python-client==1.8.0 &&
pip install dm-haiku==0.0.5 &&
pip install einops==0.3.0 &&
pip install transformers &&
pip install git+https://github.com/EleutherAI/lm-evaluation-harness/
"
gcloud compute tpus tpu-vm ssh "$3" --zone="$2" --worker=all --command "
if test -d mesh-transformer-jax; then echo \"Already checked out\"; else git clone https://github.com/nooney/mesh-transformer-jax.git; fi 
"

# Run training. TODO: Can we add some output to track progress?
gcloud compute tpus tpu-vm ssh "$3" --zone="$2" --worker=all --command "
cd mesh-transformer-jax &&
python3 device_train.py --config=configs/pmqs_config.json --tune-model-path=gs://gpt-bbc/step_383500/
"

# Check results are saved to storage

# Stop TPU
echo "Stopping TPU: $3"
# gcloud compute tpus tpu-vm describe "$3" --zone="$2"
gcloud compute tpus tpu-vm stop "$3" --zone="$2"
# Delete TPU
# echo "Deleting TPU: $3"
# gcloud compute tpus tpu-vm delete "$3" --zone="$2"
# # Start Docker daemon in TPU VM
# gcloud compute tpus tpu-vm ssh "$3" --zone="$2" --worker=all --command "sudo systemctl start docker"
# # Start Docker container
# gcloud compute tpus tpu-vm ssh "$3" --zone="$2" --worker=all --command "sudo docker run -ti --rm --name $6 --privileged --network=host python:3.8 bash"