import json
import threading
import time
from collections import deque

from nerfstudio_renderer.renderer import *


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
                'width': int,                       # The rendered image width (in pixels)
                'height': int,                      # The rendered image height (in pixels)
                'fov': float,                       # The vertical field-of-view of the camera
            }
        """
        self.cameras = cameras_config

    def default_config():
        """
        Returns a default configuration, where there are 3 cameras,
        two for accelerated and estimated rendering, and the other for
        high-resolution display.

        Returns
        ----------
        RendererCameraConfig
            A default config.
        """
        # These configurations are chosen empirically, and may be subject to change.
        # Nerfstudio camera defaults:
        # - vertical FoV: 50 degrees
        # Isaac Sim camera defaults:
        # - Size: 1280x720
        # - Focal Length: 18.14756
        # - Horizontal Aperture: 20.955
        # - Vertical Aperture: (Value Unused)
        # - (Calculated) horizontal FoV = math.degrees(2 * math.atan(20.955 / (2 * 18.14756))) = 60
        # The following vertical FoV is chosen to follow the Isaac Sim camera defaults.
        # Some useful equations:
        # - focal_length = width / (2 * math.tan(math.radians(fov_horizontal) / 2))
        # - focal_length = height / (2 * math.tan(math.radians(fov_vertical) / 2))
        # - fov_vertical = math.degrees(2 * math.atan(height / (2 * focal_length)))
        # - fov_horizontal = math.degrees(2 * math.atan(width / (2 * focal_length)))
        # - fov_horizontal = math.degrees(2 * math.atan(horiz_aperture / (2 * focal_length)))
        #   Ref: https://forums.developer.nvidia.com/t/change-intrinsic-camera-parameters/180309/6
        # - aspect_ratio = width / height
        # - fov_vertical = math.degrees(2 * math.atan((height / width) * math.tan(math.radians(fov_horizontal) / 2)))
        return RendererCameraConfig([
            # fov_vertical = math.degrees(2 * math.atan((height / width) * math.tan(math.radians(fov_horizontal) / 2)))
            # = 35.98339777135764
            # 0.05x resolution
            { 'width': 64,  'height': 36,  'fov': 35.98339777135764 },
            # 0.1x resolution
            { 'width': 128, 'height': 72,  'fov': 35.98339777135764 },
            # 0.25x resolution
            { 'width': 320, 'height': 180, 'fov': 35.98339777135764 },
            # 0.5x resolution
            { 'width': 640, 'height': 360, 'fov': 35.98339777135764 },
            # 1x resolution
            { 'width': 1280, 'height': 720, 'fov': 35.98339777135764 },
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
                 checkpoint_path,
                 device,
                 thread_count=3,
                 camera_config_path=None):
        """
        Parameters
        ----------
        model_config_path : str
            The path to model configuration .yml file.

        camera_config_path : str, optional
            The path to the config file.
            Uses `RendererCameraConfig.default_config()` when not assigned.
        """
        # Construct camera config and renderer
        self.camera_config = RendererCameraConfig.load_config(camera_config_path)
        self.renderer = NerfStudioRenderer(model_config_path, checkpoint_path, device)

        # Data maintained for optimization:
        self._last_request_camera_position = (-np.inf, -np.inf, -np.inf)
        """The camera position of the last accepted request."""
        self._last_request_camera_rotation = (-np.inf, -np.inf, -np.inf)
        """The camera rotation of the last accepted request."""

        self._request_deque = deque(maxlen=thread_count)
        """The queue/buffer of render requests. Since we want to drop
        stale requests/responses, the max size of the deque is simply
        set as the thread count. The deque acts like a request buffer
        instead of a task queue, which drops older requests when full.
        """
        self._request_deque_pop_lock = threading.Lock()
        """The lock for the request deque. Although deque is
        thread-safe, we still need to lock it when popping the deque
        while empty to create blocking behavior.
        """

        self._last_request_timestamp = time.time()
        """The timestamp of the last accepted request."""
        self._last_request_timestamp_lock = threading.Lock()
        """The timestamp lock for the last request timestamp."""
        self._last_response_timestamp = time.time()
        """The timestamp of the last sent response."""
        self._last_response_timestamp_lock = threading.Lock()
        """The timestamp lock for the last response timestamp."""

        self._image_response_buffer = None
        """The latest rendered image buffer, which will be cleared
        immediately after retrieval."""
        self._image_response_buffer_lock = threading.Lock()
        """The image lock for the image response buffer."""

        for i in range(thread_count):
            t = threading.Thread(target=self._render_task)
            t.daemon = True
            t.start()
        # We choose to use threading here instead of multiprocessing
        # due to lower overhead. We are aware of the GIL, but since
        # the bottleneck should lie in the rendering process, which
        # is implemented in C++ by PyTorch, the GIL should be released
        # during PyTorch function calls.
        # Ref: https://discuss.pytorch.org/t/can-pytorch-by-pass-python-gil/55498
        # After going through some documents, we conclude that switching
        # to multiprocessing may not be a good idea, since the overhead
        # of inter-process communication may be high, and the
        # implementation is not trivial.

    def get_rgb_image(self):
        """
        Retrieve the most recently ready rgb image.
        If no rgb images have been rendered since last call of `get_rgb_image`, returns None.

        Returns
        ----------
        np.array or None
            If applicable, returns an np array of size (width, height, 3) and with values ranging from 0 to 1.
            Otherwise, returns None.
        """
        with self._image_response_buffer_lock:
            image = self._image_response_buffer
            self._image_response_buffer = None
            return image

    def update_camera(self, position, rotation):
        """
        Notifies an update to the camera pose.
        This may or may not result in a new render request.

        Parameters
        ----------
        position : list[float]
            A 3-element list specifying the camera position.

        rotation : list[float]
            A 3-element list specifying the camera rotation, in euler angles.
        """
        if self._is_input_similar(position, rotation):
            return
        self._last_request_camera_position = position.copy()
        self._last_request_camera_rotation = rotation.copy()
        now = time.time()
        with self._last_request_timestamp_lock:
            self._last_request_timestamp = now

        # Queue this render request, with request timestamp attached.
        self._request_deque.append((position, rotation, now))

    def _render_task(self):
        while True:
            with self._request_deque_pop_lock:
                if len(self._request_deque) == 0:
                    time.sleep(0.05)
                    continue
                task = self._request_deque.pop()
            position, rotation, timestamp = task
            # For each render request, render lower quality images first, and then higher quality ones.
            # This rendering request and response may be dropped, as newer requests/responses invalidate older ones.
            for camera in self.camera_config.cameras:
                # A request can be invalidated if there are newer requests.
                with self._last_request_timestamp_lock:
                    if timestamp - self._last_request_timestamp < 0:
                        continue
                # Render the image
                # TODO: Allow early return if the request is invalidated.
                image = self.renderer.render_at(position, rotation, camera['width'], camera['height'], camera['fov'])
                # A response must be dropped if there are newer responses.
                with self._last_response_timestamp_lock:
                    if timestamp - self._last_response_timestamp < 0:
                        continue
                    self._last_response_timestamp = timestamp
                with self._image_response_buffer_lock:
                    self._image_response_buffer = image

    # Checks if camera pose is similar to what was recorded.
    def _is_input_similar(self, position, rotation):
        return position == self._last_request_camera_position and rotation == self._last_request_camera_rotation
