#!/bin/bash
# This is a comment!
{
	virtualenv -p python3 $1 &&
	. $1/bin/activate &&
	pip install -r scripts/requirements.txt

} || {
	set -e
}
deactivate
# rm -rf $1