import json
import pathlib
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
from pathlib import Path
from utils import *

class NerfStudioRenderer():
	"""
	The class is responsible for giving rendered images,
	given the position, rotation, width, height, and fov
	of a camera.
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