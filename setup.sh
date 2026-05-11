#!/bin/bash
# Update package list
sudo apt update

# Install essential build tools and dependencies
sudo apt install -y python3-distutils git-lfs build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
  libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget \
  libbz2-dev curl make llvm xz-utils tk-dev libxml2-dev libxmlsec1-dev  liblzma-dev

git lfs install --local
git lfs pull

# Install pyenv
# curl https://pyenv.run | bash

# Add to shell configuration (for bash)
# cat >> ~/.bashrc << 'EOF'
#   export PYENV_ROOT="$HOME/.pyenv"
#   export PATH="$PYENV_ROOT/bin:$PATH"
#   eval "$(pyenv init -)"
#   EOF

# Restart shell or source
# source ~/.bashrc

# Verify pyenv is installed
# pyenv --version

# Install Python 3.12.7 (this takes time)
# pyenv install 3.12.7

# Set local Python version for this directory
# pyenv local 3.12.7

# Create virtual environment
python -m venv venv3.12.7

# Activate virtual environment
source venv3.12.7/bin/activate

# upgrade pip
python -m pip install --upgrade pip

# Install from requirements file
pip install -r requirements.txt

# incase of failure
pip install -r easyReqs.txt
