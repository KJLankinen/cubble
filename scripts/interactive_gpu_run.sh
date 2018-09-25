#!/bin/bash
sinteractive --gres=gpu:1 --constraint="pascal" --time=04:00:00 --mem=24G
