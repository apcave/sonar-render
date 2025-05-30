#!/bin/bash

sudo docker build -t sonar-render:latest .

docker run --gpus all --rm -it --privileged sonar-render:latest

