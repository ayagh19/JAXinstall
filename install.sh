#! /bin/bash

source setup.sh

rm -rf venv_project

python3 -m venv --system-site-packages venv_project

source venv_project/bin/activate

pip install -U pip

pip install intel-extension-for-openxla

pip install -r https://raw.githubusercontent.com/intel/intel-extension-for-openxla/main/test/requirements.txt
