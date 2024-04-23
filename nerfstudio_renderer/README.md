# Nerfstudio Renderer

The following instructions assume you are in the `/nerfstudio_renderer` directory under the git repository root:

```sh
git clone https://github.com/j3soon/omni-nerf-extension.git
cd omni-nerf-extension
cd nerfstudio_renderer
```

## Launching NeRF Studio

> You can skip this section if you want to download the example poster model checkpoint and mesh.

Follow the [installation](https://docs.nerf.studio/quickstart/installation.html#use-docker-image) guide, specifically:

```sh
mkdir data
docker run --rm -it --gpus all \
  -u $(id -u) \
  -v $(pwd)/data:/workspace/ \
  -v $HOME/.cache/:/home/user/.cache/ \
  -p 7007:7007 \
  --shm-size=12gb \
  dromni/nerfstudio:0.3.4
```

The following subsections assume you have launched the container and using its interactive shell.

### Training a NeRF Model

Follow the [training model](https://docs.nerf.studio/quickstart/first_nerf.html) guide, specifically:

```sh
# in the container
# Download some test data:
ns-download-data nerfstudio --capture-name=poster
# Train model without normal prediction (used in the provided example poster assets for simplicity)
ns-train nerfacto --data data/nerfstudio/poster
# or train model with normal prediction (preferred)
ns-train nerfacto --data data/nerfstudio/poster --pipeline.model.predict-normals True
# wait for training to finish
```

> If you have trouble downloading the dataset, please refer to [this pull request](https://github.com/nerfstudio-project/nerfstudio/pull/3045).

### View the NeRF Model

```sh
# in the container
# change the DATE_TIME to the actual value
DATE_TIME=2023-12-30_111633
# View model
ns-viewer --load-config outputs/poster/nerfacto/$DATE_TIME/config.yml
# open the printed URL
```

### Exporting a Mesh

Follow the [export geometry](https://docs.nerf.studio/quickstart/export_geometry.html) guide, specifically:

```sh
# in the container
# change the DATE_TIME to the actual value
DATE_TIME=2023-12-30_111633
# Export mesh
# center is (-0.2, 0.1, -0.2)
ns-export tsdf --load-config outputs/poster/nerfacto/$DATE_TIME/config.yml --output-dir exports/mesh/ --target-num-faces 50000 --num-pixels-per-side 2048 --use-bounding-box True --bounding-box-min -0.55 -0.25 -0.55 --bounding-box-max 0.15 0.45 0.15
```

> Or use [Poisson Surface Reconstruction](https://docs.nerf.studio/quickstart/export_geometry.html#poisson-surface-reconstruction) instead, if the network supports predicting normals.

### View the Mesh

Open the mesh (`mesh.obj`) in Blender or any other 3D viewer.

## Download Model Checkpoint and Mesh

> You can skip this section if you want to train the example poster model checkpoint and extract mesh by yourself.

(TODO: Add link to a download a pre-trained model in release)

## Rename the Model Directory and the Checkpoint File

Rename the timestamp and checkpoint files to the same name as the placeholder for simplicity:

```sh
# change the DATE_TIME to the name of the placeholder
DATE_TIME=2023-12-30_111633
CHECKPOINT_NAME=step-000029999
cp -r ./data/outputs/poster/nerfacto/$DATE_TIME ./data/outputs/poster/nerfacto/DATE_TIME
mv ./data/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/$CHECKPOINT_NAME.ckpt ./data/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt
```

You can check if the renaming succeeded with the following commands:

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
> seems to fix the issue (`docker compose down` isn't enough). I believe
it is due to the `pip install` in docker entrypoint. Please keep this in
> mind when modifying the renderer code.

For development purposes, you can run the following command to run the
PyGame test window directly in the `nerfstudio-renderer` container:

```sh
docker compose build
xhost +local:docker
docker compose up
# in new shell
docker exec -it nerfstudio-renderer /workspace/tests/run_local.sh
```

The `run_local.sh` script will re-copy and re-install the package
before launching the PyGame window, so this method will not encounter
the old code issue mentioned above.

## Running Inside Docker

Alternatively, it is possible to connect to the server with [rpyc](https://github.com/tomerfiliba-org/rpyc) in the `pygame-window` container.

```python
import rpyc
import random
import time

# Make connection
conn = rpyc.classic.connect('localhost', port=10001)

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

Please note that the use of rpyc does not perfectly decouple the client and server. The client must be using the same Python version as the server, otherwise, there will be compatibility issues.

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
