#!/bin/bash

sudo docker build -t sonar-render:latest .

docker run --gpus=all --runtime=nvidia --rm --privileged sonar-render:latest

