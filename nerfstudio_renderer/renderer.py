import json
import pathlib
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
from pathlib import Path
from utils import *

class RendererCameraConfig:
	"""
	This class contains functions used to load
	camera configurations for the NerfStudioRenderer to use.

	The configuration is a list of dicts.
	The NerfStudioRenderer is then able to render differently
	sized images with respect to each configuration, 
	for performance considerations for example.
	"""

	def __init__(self, cameras_config):
		"""
		Parameters
		----------
		cameras_config : list[dict]
			A list of dicts that describes different camera configurations.
			Each element is of the form { 'width': int, 'height': int, 'fov': float }
		"""
		self.cameras = cameras_config

	def default_config():
		"""
		Returns a default configuration, where there are 2 cameras,
		one for accelerated and estimated rendering, and another for 
		high-resolution display.

		Returns
		----------
		RendererCameraConfig
			A default config.
		"""
		return RendererCameraConfig([
			{ 'width': 90,  'height': 42,  'fov': 72 },
			{ 'width': 900, 'height': 420, 'fov': 72 }
		])

	def load_config(file_path=None):
		"""
		Returns a configuration defined by a json-formatted file.
		
		Parameters
		----------
		file_path : str, optional
			The path to the config file.

		Returns
		----------
		RendererCameraConfig
			A config specified by `file_path`, or a default one.
		"""
		if file_path is None:
			return RendererCameraConfig.default_config()
		with open(file_path, 'r') as f:
			return RendererCameraConfig(json.load(f))
		
	def __len__(self):
		"""
		Returns
		----------
		int
			The number of cameras in this configuration list.
		"""
		return len(self.cameras)
	
	def __getitem__(self, idx):
		"""
		Returns
		----------
		dict
			The camera configuration indexed by `idx`.
		"""
		return self.cameras[idx]

class NerfStudioRenderer():
	"""
	The class is responsible for giving rendered images,
	given a `RendererCameraConfig`, position, rotation, 
	and a desired quality bound.
	"""

	def __init__(self, model_config_path, eval_num_rays_per_chunk=None):
		"""
		Parameters
		----------
		model_config_path : Path
			The path to model configuration .yml file.

		eval_num_rays_per_chunk : int, optional
			The parameter `eval_num_rays_per_chunk` to pass to `nerfstudio.utils.eval_utils.eval_setup`
		"""
		self.model_config, self.pipeline, _, _ = eval_setup(
            model_config_path,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
            test_mode='inference',
        )

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
			The field-of-view of the camera.

		Returns
		----------
		np.array
			An np array of rgb values.
		"""
		# Obtain a Cameras object, and transform it to the same device as the pipeline.
		c2w_matrix = camera_to_world_matrix(position, rotation)
		cameras = create_cameras(c2w_matrix, width, height, fov).to(self.pipeline.device)

		# Obtain a ray bundle with this Cameras
		ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)

		# Inference
		with torch.no_grad():
			outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)['rgb']

		# Return results
		return outputs.cpu().numpy()