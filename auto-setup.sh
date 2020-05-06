#!/bin/bash
pip3 install virtualenv
echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc
cd $HOME/EchoCount
virtualenv venv
source venv/bin/activate
pip3 install -r requirements-gpu.txt
mkdir $HOME/screenlogs
screen -d -m -S training -L -Logfile $HOME/screenlogs/training.log
screen -d -m -S tensorboard -L -Logfile $HOME/screenlogs/tensorboard.log
