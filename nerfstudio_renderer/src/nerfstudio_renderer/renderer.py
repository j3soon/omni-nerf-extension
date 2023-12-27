import torch
import yaml
from nerfstudio.data.scene_box import SceneBox
from nerfstudio_renderer.utils import *


class NerfStudioRenderer():
    """
    The class is responsible for giving rendered images,
    given the position, rotation, width, height, and fov
    of a camera.
    """

    def __init__(self, model_config_path, checkpoint_path, device):
        """
        Parameters
        ----------
        model_config_path : Path
            The path to model configuration .yml file.

        checkpoint_path : Path or str
            The path to model checkpoint .ckpt file.

        device : torch.device
            Device for the model to run on. Usually CUDA or CPU.
        """
        # Load checkpoint
        loaded_state = torch.load(checkpoint_path, map_location="cpu")
        loaded_state, step = loaded_state["pipeline"], loaded_state["step"]

        # Assert all-zero appearance embedding
        config = yaml.load(model_config_path.read_text(), Loader=yaml.Loader)
        config.pipeline.model.use_average_appearance_embedding = False

        # Gather model-related arguments
        scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32))
        metadata = { }
        grad_scaler = None
        num_train_data = 0

        self.model = config.pipeline.model.setup(
            scene_box=scene_box,
            num_train_data=num_train_data,
            metadata=metadata,
            device=device,
            grad_scaler=grad_scaler,
        ).to(device)

        # Update model to step
        self.model.update_to_step(step)

        # Alter loaded model state dict for loading
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }

        is_ddp_model_state = True
        model_state = {}
        for key, value in state.items():
            if key.startswith("_model."):
                model_state[key[len("_model.") :]] = value
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False

        # Drop appearance embedding (training only)
        model_state = { key: value for key, value in model_state.items() if 'embedding_appearance' not in key }

        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = { key[len("module.") :]: value for key, value in model_state.items() }

        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()

        # ---------
        self.device = device

    def render_at(self, position, rotation, width, height, fov):
        """
        Parameters
        ----------
        position : list[float]
            A 3-element list specifying the camera position.

        rotation : list[float]
            A 3-element list specifying the camera rotation, in euler angles.

        width : int
            The width of the camera.

        height : int
            The height of the camera.

        fov : float
            The vertical field-of-view of the camera.

        Returns
        ----------
        np.array
            An np array of rgb values.
        """
        # Obtain a Cameras object, and transform it to the same device as the model.
        c2w_matrix = camera_to_world_matrix(position, rotation)
        cameras = create_cameras(c2w_matrix, width, height, fov).to(self.device)

        # Obtain a ray bundle with this Cameras
        ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)

        # Inference
        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)['rgb']

        # Return results
        return outputs.cpu().numpy()