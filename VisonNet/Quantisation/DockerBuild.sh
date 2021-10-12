#!/bin/bash
# This is script which creates docker image used in the Quantisation of the model
# For Linux
sudo DOCKER_BUILDKIT=1 docker build -t visonquantimage -f Dockerfile ..
