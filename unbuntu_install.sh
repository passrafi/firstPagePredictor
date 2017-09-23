#!/bin/bash

sudo apt-get install python-pip python-dev python-virtualenv 
virtualenv --system-site-packages ~/tensorflow
source ~/tensorflow/bin/activate
easy_install -U pip
pip install --upgrade tensorflow-gpu 
python -m nltk.downloader wordnet
