#!/usr/bin/env bash

# Install necessary package
sudo apt update && sudo apt install unrar

# Download and extract the dataset
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar && \
    unrar x UCF101.rar
