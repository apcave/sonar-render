#!/bin/bash

sudo docker build -t sonar-render:latest .

sudo docker run --gpus=all --runtime=nvidia --rm -it --privileged sonar-render:latest

