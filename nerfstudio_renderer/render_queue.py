import time
import threading
from renderer import *

class RendererCameraConfig:
	"""
	This class contains functions used to load
	camera configurations for the NerfStudioRenderQueue to use.

	The configuration is a list of dicts.
	The NerfStudioRenderQueue is then able to render differently
	sized images with respect to each configuration, 
	for performance considerations for example.
	"""

	def __init__(self, cameras_config):
		"""
		Parameters
		----------
		cameras_config : list[dict]
			A list of dicts that describes different camera configurations.
			Each element is of the form { 
                'width': int,                       # The targeted rendered width
                'height': int,                      # The targeted rendered height
                'fov': float,                       # The targeted rendered height
                'num_allowed_render_calls': int,    # The maximum number of render calls allowed for this configuration
                'delay_before_render_call': int     # The delay before making a render call for this configuration
            }
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
			{ 'width': 90,  'height': 42,  'fov': 72, 'num_allowed_render_calls': 5, 'delay_before_render_call': 0   },
			{ 'width': 900, 'height': 420, 'fov': 72, 'num_allowed_render_calls': 2, 'delay_before_render_call': 0.1 }
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

class NerfStudioRenderQueue():
    """
	The class encapsulates NerfStudioRenderer and provides
    a mechanism that aims at minimizing rendering latency,
    via an interface that allows registration of rendering
    requests. The render queue attempts to deliver 
    rendering results of the latest request in time, so 
    requests are not guaranteed to be served.

    Attributes
    ----------
	camera_config : RendererCameraConfig
        The different configurations of cameras (different qualities, etc.).

    renderer : NerfStudioRenderer
        The NerfStudioRenderer used to actually give rendered images.
	"""
    
    def __init__(self, 
                 model_config_path, 
                 camera_config_path=None, 
                 eval_num_rays_per_chunk=None, 
                 pose_check_position_threshold=0.01,
                 pose_check_rotation_threshold=5):
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

        pose_check_position_threshold : float, optional
            Two cameras are treated as identical in position, 
            if the sum of squared differences of their position vectors is under this threshold.

        pose_check_position_threshold : float, optional
            Two cameras are treated as identical in rotation, 
            if the sum of differences of their euler angle rotation vectors is under this threshold.
        """
        # Construct camera config and renderer
        self.camera_config = RendererCameraConfig.load_config(camera_config_path)
        self.renderer = NerfStudioRenderer(model_config_path, eval_num_rays_per_chunk)

        # Pose Check Thresholds
        self._pose_check_position_threshold = pose_check_position_threshold
        self._pose_check_rotation_threshold = pose_check_rotation_threshold

        # Data maintained for optimization:
        # The camera position of the most recently accepted request.
        self._recent_camera_position = (-np.inf, -np.inf, -np.inf)
        # The camera rotation of the most recently accepted request.
        self._recent_camera_rotation = (-np.inf, -np.inf, -np.inf)
        # The most recently completed request id
        self._recent_complete_request_id = 0
        # The most recently accepted request id
        self._recent_accepted_request_id = -1
        # The data lock for avoiding race conditions
        self._data_lock = threading.Lock()
        # The semaphores for preventing intense request bursts.
        self._semaphores_by_quality = [threading.Semaphore(config['num_allowed_render_calls']) 
                                       for config in self.camera_config]

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
        # Optimization: Pose Check
        # If this upcoming request has the (almost) same camera pose: position and rotation
        # with the most recently accepted request, ignore it.
        if self._is_pose_check_failed(position, rotation):
              return
        self._recent_camera_position = (position[0], position[1], position[2])
        self._recent_camera_rotation = (rotation[0], rotation[1], rotation[2])

        # Increment the most recently accepted request id by 1
        with self._data_lock:
            self._recent_accepted_request_id += 1

        # Start a thread of this render request, with request id attached.
        renderer_call_args = (self._recent_accepted_request_id, position, rotation, callback)
        thread = threading.Thread(target=self._progressive_renderer_call, args=renderer_call_args)
        thread.start()
	
    def _progressive_renderer_call(self, request_id, position, rotation, callback):
        # For each render request, try to deliver the render output of the lowest quality fast.
        # When rendering of lower qualities are done, serially move to higher ones.
        for quality_index, config_entry in enumerate(self.camera_config):
            # For each config of different quality: obtain the rendered image, and then call the callback.

            # Optimization: Delay Before Call
            # Apply a small delay before calls (and checks-before-calls), especially costly ones.
            # This prevents a costly call from occupying the computation resources (usually GPUs) too early,
            # as newer requests can invalidate this request with less costly calls.
            # Intuitively, only when the camera stays at a place for very long (longer than the delay) 
            # that we can confidently start costly, high-quality calls.
            # If after the delay, new requests come in, this older request will be invalidated in check-before-call.
            delay_before_render_call = config_entry['delay_before_render_call']
            if delay_before_render_call > 0:
                time.sleep(delay_before_render_call)

            # Optimization: Semaphores By Quality
            # If an intense request burst happens, a series of costly calls can clog up computation resources really fast.
            # The thread of a request can be blocked, if too many requests are already running calls of the same quality index.
            # Say a bunch of high-quality, costly calls are blocked, because some have made costly calls but have not obtained results.
            # When a vacancy is available, these blocked calls may proceed, but most of them will be invalidated by check-before-call.
            # Among the many blocked costly calls, only the one from the newest request may proceed.
            # Intuitively, we try to select newer requests to make costly calls with this optimization.
            self._semaphores_by_quality[quality_index].acquire()

            # Optimization: Check Before Call
            # A request can be invalidated before a call of it is made,
            # if there are newer requests accepted.
            with self._data_lock:
                if request_id < self._recent_accepted_request_id:
                    self._semaphores_by_quality[quality_index].release()
                    return

            # Render the image
            image = self.renderer.render_at(position, rotation, config_entry['width'], config_entry['height'], config_entry['fov'])

            # Release the semaphore acquired from semaphores-by-quality after the render call is done.
            self._semaphores_by_quality[quality_index].release()

            # Optimization: Check After Call
            # When a call is finished, its results may no longer be needed (i.e., obsolete)
            # Maintain the most recent request id that completed (some of) its render calls.
            # If a newer request has finished a call before this request, discard the results.
            # Using completed request id (instead of accepted request id) prevents the situation where no results are accepted.
            with self._data_lock:
                if request_id < self._recent_complete_request_id:
                    return
                else:
                    self._recent_complete_request_id = request_id
            
            # Callback
            callback(image)

    def _is_pose_check_failed(self, position, rotation):
          position_diff = sum([(a - b) * (a - b) for a, b in zip(position, self._recent_camera_position)])
          rotation_diff = sum([(a - b) for a, b in zip(rotation, self._recent_camera_rotation)])
          return (position_diff >= self._pose_check_position_threshold) and (rotation_diff >= self._pose_check_rotation_threshold)