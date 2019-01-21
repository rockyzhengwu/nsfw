#!/usr/bin/env bash

MODEL_NAME=nsfw
MODEL_BASE_PATH=`pwd`/data/models
tensorflow_model_server --port=8600 --rest_api_port=8601  --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}