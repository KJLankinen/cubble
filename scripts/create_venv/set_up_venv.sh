#!/bin/bash
# Creates new virtual python environment, activates it and installs required packages
{
	python3 -m venv $1 &&
	. $1/bin/activate &&
	pip install -r $2

} || {
	set -e
}
deactivate
# rm -rf $1