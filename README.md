# Omniverse NeRF Plugin

## Prerequisites

- **Hardware**:
  - CPU: x86
  - GPU: NVIDIA RTX GPU
  - See [this page](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html#system-requirements) for more details.
- **Operating System**: Ubuntu 20.04/22.04.
- **Software**:
  - [NVIDIA Driver](https://ubuntu.com/server/docs/nvidia-drivers-installation)
  - [Docker](https://docs.docker.com/engine/install/ubuntu/)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Setup

```sh
git clone https://github.com/j3soon/omni-nerf-extension
cd omni-nerf-extension
```

Download assets:

```sh
wget https://github.com/j3soon/omni-nerf-extension/releases/download/v0.0.1/assets.zip
unzip assets.zip
```

Prepare assets for `nerfstudio_renderer`:

```sh
# change the DATE_TIME to the name of the placeholder
DATE_TIME=2023-12-30_111633
CHECKPOINT_NAME=step-000029999
cp -r ./assets/outputs/poster/nerfacto/$DATE_TIME ./assets/outputs/poster/nerfacto/DATE_TIME
mv ./assets/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/$CHECKPOINT_NAME.ckpt ./assets/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt
```

The following assumes that you are running the commands from the root of the repository.

## Managing Containers

Login to NGC and pull the image `nvcr.io/nvidia/isaac-sim:2023.1.1` by following [this guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html). Then build the docker images for the extension:

```sh
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
docker compose build
```

Launch the containers:

```sh
# You might want to use `tmux` for exec-ing into the containers later
xhost +local:docker
docker compose up
```

Then follow the remaining sections.

To remove and stop the containers, run:

```sh
docker compose down
```

### Nerfstudio Renderer

Code: [`nerfstudio_renderer`](./nerfstudio_renderer)

The renderer server would be listening on port `10001` upon successful startup:

```
INFO SLAVE/10001[MainThread]: server started on [0.0.0.0]:10001
```

After seeing the above logs, no additional steps are required for the renderer server.

### PyGame Viewer

Code: [`pygame_viewer`](./pygame_viewer)

Attach to the container and run the testing script:

```sh
docker exec -it pygame-viewer /src/run.sh
```

The script may fail at the first run due to the cold start of the renderer server. If it fails, try run the script again.

(TODO: Preview Video)

### Isaac Sim Viewer

Code: [`extension`](./extension)

```sh
docker exec -it isaac-sim-viewer bash
# in container
/isaac-sim/runapp.sh --ext-folder /src/exts --enable omni.nerf.viewport
```

(TODO: Preview Video x2)

## Development Notes

### Nerfstudio Renderer

After modifying code, you need to remove and recreate the container to apply changes. This is because the container will copy and install the code upon startup.

### PyGame Viewer

After modifying code, you need to re-run the testing script. The docker container can be re-used since the code is mounted as a volume.

### Isaac Sim Viewer

Setup VSCode intellisense for the extension:

```sh
cd extension
./link_app.sh --path "$HOME/.local/share/ov/pkg/code-2022.3.3"
# open the `extension` folder in VSCode
```

After modifying code, you can restart Isaac Sim to apply changes. The docker container can be re-used since the code is mounted as a volume. If the change is small, it is often faster to disable and re-enable the extension in the Isaac Sim UI. This can be done through `Window > Extensions > NVIDIA > General`, search `nerf`, and then un-toggle and re-toggle the extension.
