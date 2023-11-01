import time
from .renderer import *

class NerfStudioRenderQueue():
    """
	The class encapsulates NerfStudioRenderer and provides
    a mechanism that aims at minimizing rendering latency,
    via an interface that allows registration of rendering
    requests. Each request may or may not be served. 
    The render queue attempts to deliver rendering results
    of the latest request in time.

    Attributes
    ----------
    recent_camera_position : tuple[float]
        Represents the most recently rendered camera position.

    recent_camera_rotation : tuple[float]
        Represents the most recently rendered camera rotation.

    renderer : NerfStudioRenderer
        The NerfStudioRenderer used to actually give rendered images.
	"""
    
    def __init__(self, model_config_path, camera_config_path=None, eval_num_rays_per_chunk=None):
        """
        Parameters
        ----------
        model_config_path : str
            The path to model configuration .yml file.

        camera_config_path : str, optional
            The path to the config file. 
			Uses `RendererCameraConfig.default_config()` when not assigned.

        eval_num_rays_per_chunk : int, optional
            The parameter `eval_num_rays_per_chunk` to pass to `nerfstudio.utils.eval_utils.eval_setup`
        """
        self.recent_camera_position = ()
        self.recent_camera_rotation = ()
        camera_config = RendererCameraConfig.load_config(camera_config_path)
        self.renderer = NerfStudioRenderer(model_config_path, camera_config, eval_num_rays_per_chunk)

    def register_render_request(self, position, rotation, callback):
        """
        Registers a request to render with NerfStudioRenderer.

        Parameters
        ----------
        position : list[float]
            A 3-element list specifying the camera position.

        rotation : list[float]
            A 3-element list specifying the camera rotation, in euler angles.

        callback : function(np.array)
            A callback function to call when the renderer finishes this request.
        """
        pass
