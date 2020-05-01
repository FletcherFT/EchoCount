#!/bin/bash
git clone https://github.com/FletcherFT/EchoCount.git -b aws
pip3 install virtualenv
echo "export PATH=$PATH:$HOME/.local/bin"
cd EchoCount
virtualenv venv
source venv/bin/activate
pip3 install -r requirements-gpu.txt
