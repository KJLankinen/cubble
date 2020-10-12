#!/bin/bash
sinteractive --gres=gpu:1 --constraint='volta' --time=08:00:00 --mem=32G
