import json
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path

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
	
	def __item__(self, idx):
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

	def __init__(self, model_config, camera_config_path=None, eval_num_rays_per_chunk=None):
		"""
		Parameters
		----------
		model_config : Path
			The path to model configuration .yml file.

		camera_config_path : str, optional
			The path to the config file. 
			Uses `RendererCameraConfig.default_config()` when not assigned.

		eval_num_rays_per_chunk : int, optional
			The parameter `eval_num_rays_per_chunk` to pass to `nerfstudio.utils.eval_utils.eval_setup`
		"""
		self.camera_config = RendererCameraConfig.load_config(camera_config_path)
		self.model_config, self.pipeline, _, _ = eval_setup(
            model_config,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
            test_mode='inference',
        )

	def render_at(quality_bound, position, rotation):
		"""
		Parameters
		----------
		quality_bound : int
			The desired rendering quality. The renderer attempts to
			give rendered images with at least the specified quality.

		position : list[float]
			The 3-element list specifying the camera position.

		rotation : list[float]
			The 3-element list specifying the camera rotation, in euler angles.

		Returns
		----------
		np.array
			An np array of rgb values.
		"""
		pass