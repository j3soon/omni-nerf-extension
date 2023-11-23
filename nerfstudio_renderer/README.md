# Nerfstudio Renderer

The following instructions assume you are in the `/nerfstudio_renderer` directory under the git repository root.

## Prepare a NeRF Model Checkpoint

Train a NeRF and store the weights of the `poster` scene in `./data`. You can check it with the following command:

```sh
ls ./data/outputs/poster/nerfacto/<DATE_TIME>/config.yml
```

## Docker Setup

Build the dockerfile with server port `7007`.

```sh
docker build \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg OS_VERSION=22.04 \
  --build-arg SERVER_PORT=7007 \
  --tag nerfstudio-renderer-86 .
```

And run it:

```sh
docker run \
  --name nerfstudio-renderer \
  --gpus all \
  --rm \
  -v $(pwd)/data/outputs:/workspace/outputs \
  -p 7007:7007 \
  --shm-size=6gb \
  nerfstudio-renderer-86
```

## Running Inside Docker

Upon success, it is possible to connect to the server with [rpyc](https://github.com/tomerfiliba-org/rpyc).

```python
import rpyc
import random
import time

# Make connection
conn = rpyc.classic.connect('localhost', port=7007)

# Imports
conn.execute('import nerfstudio_renderer')
conn.execute('from pathlib import Path')
conn.execute('import torch')

# Create a NerfStudioRenderQueue
# For some reason, netref-based methods keep resulting in timeouts.
conn.execute('rq = nerfstudio_renderer.NerfStudioRenderQueue(model_config_path=Path("<MODEL_CONFIG_PATH>"), checkpoint_path="<MODEL_CHECKPOINT_PATH>", device=torch.device("cuda"))')

# Update camera pose
position = [random.random() for _ in range(3)]
rotation = [0., -152, 0.]
conn.execute(f'rq.update_camera({position}, {rotation})')

# Wait for some time...
time.sleep(3)

# Obtain a rendered image
image = conn.eval('rq.get_rgb_image()')

# Delete remote render queue
conn.execute('del rq')
```

## Running with A Test Script (Pygame)

Outside the container, run:

```sh
pip3 install -r requirements.txt
# python3 pygame_test.py --model_config_path=<MODEL_CONFIG_PATH_IN_DOCKER>
python3 pygame_test.py --model_config_path=/workspace/outputs/poster/nerfacto/<DATE_TIME>/config.yml --model_checkpoint_path=/workspace/outputs/poster/nerfacto/<DATE_TIME>/nerfstudio_models/<CHECKPOINT_NAME>.ckpt
```

## Notes

- `NerfStudioRenderQueue.update_camera` can be called whenever needed. The renderer will progressively render better images serially. Each update to the camera will result in an asynchronous rendering series.
- `NerfStudioRenderQueue.get_rgb_image` will always return a newly rendered image.
- These two calls need not to be paired.
- After a call to `NerfStudioRenderQueue.get_rgb_image`, its return value will become `None` until:
  1. Another image from a *newer* camera update is completed.
  2. Another image from the same camera update is completed, in higher quality than the previous ones, and no images from newer updates have been ready at that point.
- **No-Way-Back Guarantee**: If an image from a newer update (say, the 10-th update) is ready at `NerfStudioRenderQueue.get_rgb_image` (even if it is never retrieved), it is guaranteed no image from the 1-st to 9-th updates will be given by future calls.
  - Therefore, it is safe to call `NerfStudioRenderQueue.get_rgb_image` multiple times just to check if a newer render is done between these calls.
  - You may not immediately get newest renders, but you will never get two renders in reversed time ordering.
