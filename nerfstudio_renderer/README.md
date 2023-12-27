# Nerfstudio Renderer

The following instructions assume you are in the `/nerfstudio_renderer` directory under the git repository root.

## Prepare a NeRF Model Checkpoint

Train a NeRF and store the weights of the `poster` scene in `./data`.

For simplicity, please change the datetime and checkpoint name to the following:

```sh
./data/outputs/poster/nerfacto/DATE_TIME/config.yml
./data/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt
```

You can check it with the following command:

```sh
ls ./data/outputs/poster/nerfacto/DATE_TIME/config.yml
ls ./data/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt
```

## Running with Docker Compose

Run the PyGame test window with the following commands:

```sh
xhost +local:docker
docker compose up
# in new shell
docker exec -it pygame-window /workspace/run.sh
# the initial execution might result in a delay due to the download of the pre-trained torch model.
# please re-run the script if the script times out.
```

> There seems to be an issue in `nerfstudio-renderer` that uses old code
> upon restart. I'm not aware of a reliable fix for this issue yet.
> However, running `docker compose down && docker rm $(docker ps -aq)`
> seems to fix the issue. Please keep this in mind when modifying the
> renderer code.

## Running Inside Docker

Alternatively, it is possible to connect to the server with [rpyc](https://github.com/tomerfiliba-org/rpyc) in the `pygame-window` container.

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
conn.execute('rq = nerfstudio_renderer.NerfStudioRenderQueue(model_config_path=Path("/workspace/outputs/poster/nerfacto/DATE_TIME/config.yml"), checkpoint_path="/workspace/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt", device=torch.device("cuda"))')

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
