#!/bin/bash

# The DATE_TIME and CHECKPOINT_NAME are placeholders for the actual values
# You can rename the config.yml and checkpoint files to the same name as the placeholder for simplicity
python3 pygame_test.py \
    --model_config_path=/workspace/outputs/poster/nerfacto/DATE_TIME/config.yml \
    --model_checkpoint_path=/workspace/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt
