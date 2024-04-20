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
        # Originally, `nerfstudio.utils.eval_setup` is used to load the entire Pipeline, which takes as input a TrainerConfig yml file.
        # During the TrainerConfig setup (`nerfstudio.configs.base_config`) process, the constructor of VanillaPipeline is called.
        # It will set up several components to form a complete pipeline, including a DataManager.
        # The DataManager (VanillaDataManager) will perform operations to obtain DataParser outputs.
        # During the setup process of the DataParser (NerfstudioDataParser), an assert is made, which forces the presence of training dataset.
        # See: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/dataparsers/nerfstudio_dataparser.py#L86
        # Thus, even when performing inference, training dataset is needed.
        # The following code is a workaround that doesn't require to set up the entire Pipeline.
        # It load solely the model checkpoint with regard to its TrainerConfig YAML, without needing to set up the entire Pipeline.

        self.device = device

        # 1. Entrypoint `eval_setup`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L68

        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L88
        config = yaml.load(model_config_path.read_text(), Loader=yaml.Loader)
        # Using zero or average appearance embedding is a inference-time choice,
        # not a training-time choice (that would require inference-time to match such a choice).
        # Therefore, we simply choose to use zero appearance embedding
        # See Section B of the NeRF-W paper's supplementary material
        # Ref: https://arxiv.org/abs/2008.02268v3
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/fields/nerfacto_field.py#L247-L254
        if config.pipeline.model.use_average_appearance_embedding:
            print("WARNING: Forcing zero appearance embedding, although model config specifies to use average appearance embedding.")
        config.pipeline.model.use_average_appearance_embedding = False
        # TODO: Support configuring `eval_num_rays_per_chunk`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L92-L93

        # 1.1. Call to `VanillaPipelineConfig.setup`, which inherits `InstantiateConfig.setup`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L103
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/configs/base_config.py#L52

        # 1.2. Call to `VanillaPipelineConfig._target`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/configs/base_config.py#L54
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L224

        # 1.3. Call to `VanillaPipeline.__init__`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L224
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L251

        # 1.3.1. Call to `VanillaDataManagerConfig.setup`, which inherits `InstantiateConfig.setup`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L263-L265
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/configs/base_config.py#L54

        # 1.3.2. Call to `VanillaDataManagerConfig._target`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/configs/base_config.py#L54
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/datamanagers/base_datamanager.py#L320

        # 1.3.3. Call to `VanillaDataManager.__init__`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/datamanagers/base_datamanager.py#L320
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/datamanagers/base_datamanager.py#L378

        # 1.3.4. Call to `get_dataparser_outputs`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/datamanagers/base_datamanager.py#L403
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/dataparsers/base_dataparser.py#L155

        # 1.3.5. `_generate_dataparser_outputs`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/dataparsers/base_dataparser.py#L165
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/dataparsers/nerfstudio_dataparser.py#L85

        # Gather model-related arguments
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/dataparsers/nerfstudio_dataparser.py#L256-L263
        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = config.pipeline.datamanager.dataparser.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/data/dataparsers/nerfstudio_dataparser.py#L319-L322
        metadata={
            "depth_filenames": None, # depth filenames are only required during training
            "depth_unit_scale_factor": config.pipeline.datamanager.dataparser.depth_unit_scale_factor,
        }

        # 1.4. Call to `VanillaPipeline.setup`

        # Setting num_train_data to 0 is fine, since we are not using average appearance embedding.
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L272
        num_train_data = 0
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L275
        grad_scaler = None # only required during training
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L270-L276
        self.model = config.pipeline.model.setup(
            scene_box=scene_box,
            num_train_data=num_train_data,
            metadata=metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        # Move model to device
        self.model.to(device)

        # 2. Call to `pipeline.eval()`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L105
        self.model.eval()

        # 3. Call to `eval_load_checkpoint`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L108
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L35

        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L62
        loaded_state = torch.load(checkpoint_path, map_location="cpu")
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L63
        loaded_state, step = loaded_state["pipeline"], loaded_state["step"]

        # 4. Call to `VanillaPipeline.load_pipeline`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/utils/eval_utils.py#L63
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L423

        # Alter loaded model state dict for loading and update model to step
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L430-L433
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)

        # 5. Call to `Pipeline.load_state_dict`
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L434
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L109

        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L110-L119
        is_ddp_model_state = True
        model_state = {}
        for key, value in state.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # Drop the embedding layer for appearance embedding that requires the number of training images,
        # since we are not using average appearance embedding.
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/fields/nerfacto_field.py#L112
        model_state = { key: value for key, value in model_state.items() if 'embedding_appearance' not in key }
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L120-L122
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = { key[len("module.") :]: value for key, value in model_state.items() }
        # Ref: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/pipelines/base_pipeline.py#L130
        self.model.load_state_dict(model_state, strict=False)

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
            # TODO: Allow early return between the calculation of ray bundles if the request is invalidated.
            # See: https://github.com/nerfstudio-project/nerfstudio/blob/c87ebe34ba8b11172971ce48e44b6a8e8eb7a6fc/nerfstudio/models/base_model.py#L175
            outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)['rgb']

        # Return results
        return outputs.cpu().numpy()