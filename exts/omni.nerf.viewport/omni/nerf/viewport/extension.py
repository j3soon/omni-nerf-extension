import platform

import cv2
import numpy as np
import omni.ext
import omni.ui as ui
import omni.usd
import rpyc
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom


# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    print("[omni.nerf.viewport] some_public_function was called with x: ", x)
    return x ** x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class OmniNerfViewportExtension(omni.ext.IExt):

    def __init__(self):
        super().__init__()
        self.is_python_supported: bool = platform.python_version().startswith("3.10")
        """The Python version must match the backend version for RPyC to work."""
        self.camera_position: Gf.Vec3d = None
        self.camera_rotation: Gf.Vec3d = None

    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        # To see the Python print output in Omniverse Code, open the `Script Editor`.
        # In Isaac Sim, see the startup console instead.
        print("[omni.nerf.viewport] omni nerf viewport startup")
        self.selected_camera_path = None
        # Ref: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/stage/get-current-stage.html
        self.usd_context = omni.usd.get_context()
        # Subscribe to event streams
        # Ref: https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/event_streams.html
        # Listen to selection changes
        # Ref: https://docs.omniverse.nvidia.com/workflows/latest/extensions/object_info.html#step-3-4-use-usdcontext-to-listen-for-selection-changes
        self.stage_event_stream = self.usd_context.get_stage_event_stream()
        self.stage_event_delegate = self.stage_event_stream.create_subscription_to_pop(
            self._on_stage_event, name="Object Info Selection Update"
        )
        # TODO: Subscribe to only certain event types
        # Ref: https://docs.omniverse.nvidia.com/kit/docs/kit-manual/104.0/carb.events/carb.events.IEventStream.html#carb.events.IEventStream.create_subscription_to_pop_by_type
        # Listen to rendering events. Only triggered when the viewport is rendering is updated.
        # Will not be triggered when no viewport is visible on the screen.
        # Examples on using `get_rendering_event_stream` can be found by installing Isaac Sim
        # and searching for `get_rendering_event_stream` under `~/.local/share/ov/pkg/isaac_sim-2023.1.1`.
        self.rendering_event_stream = self.usd_context.get_rendering_event_stream()
        self.rendering_event_delegate = self.rendering_event_stream.create_subscription_to_pop(
            self._on_rendering_event, name="NeRF Viewport Update"
        )
        # TODO: Consider subscribing to update events
        # Ref: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/events.html#subscribe-to-update-events
        # Allocate memory
        self.rgba_w, self.rgba_h = 640, 480
        self.rgba = np.ones((self.rgba_h, self.rgba_w, 4), dtype=np.uint8) * 128
        """RGBA image buffer. The shape is (H, W, 4), following the NumPy convention."""
        self.rgba[:,:,3] = 255
        # Init RPyC connection
        if self.is_python_supported:
            self.init_rpyc()
        # Build UI
        self.build_ui()

    def init_rpyc(self):
        # TODO: Make the following configurable
        host = 'localhost'
        port = 10001
        model_config_path = '/workspace/outputs/poster/nerfacto/DATE_TIME/config.yml'
        model_checkpoint_path = '/workspace/outputs/poster/nerfacto/DATE_TIME/nerfstudio_models/CHECKPOINT_NAME.ckpt'
        device = 'cuda'
        self.rpyc_conn = rpyc.classic.connect(host, port)
        self.rpyc_conn.execute('from nerfstudio_renderer import NerfStudioRenderQueue')
        self.rpyc_conn.execute('from pathlib import Path')
        self.rpyc_conn.execute('import torch')
        self.rpyc_conn.execute(f'rq = NerfStudioRenderQueue(model_config_path=Path("{model_config_path}"), checkpoint_path="{model_checkpoint_path}", device=torch.device("{device}"))')

    def build_ui(self):
        """Build the UI. Should be called upon startup."""
        # Please refer to the `Omni::UI Doc` tab in Omniverse Code for efficient development.
        # Ref: https://youtu.be/j1Pwi1KRkhk
        # Ref: https://github.com/NVIDIA-Omniverse
        # Ref: https://youtu.be/dNLFpVhBrGs
        self.ui_window = ui.Window("NeRF Viewport", width=self.rgba_w, height=self.rgba_h)

        with self.ui_window.frame:
            with ui.VStack():
                self.ui_lbl = ui.Label("(To Be Updated)")
                state = "supported" if platform.python_version().startswith("3.10") else "NOT supported"
                self.ui_lbl.text = f"Python {platform.python_version()} is {state}"
                self.ui_btn = ui.Button("Reset Camera", width=20, clicked_fn=self.on_btn_click)
                # Camera Viewport
                # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.docs/latest/overview.html#simplest-example
                # Don't create a new viewport widget as below, since the viewport widget will often flicker.
                # Ref: https://docs.omniverse.nvidia.com/dev-guide/latest/release-notes/known-limits.html
                # ```
                # from omni.kit.widget.viewport import ViewportWidget
                # self.ui_viewport_widget = ViewportWidget(
                #     resolution = (640, 480),
                #     width = 640,
                #     height = 480,
                # )
                # self.viewport_api = self.ui_viewport_widget.viewport_api
                # ````
                # Ref: https://docs.omniverse.nvidia.com/dev-guide/latest/python-snippets/viewport/change-viewport-active-camera.html
                # Instead, the viewport is obtained from the active viewport in new renderings.

                # NeRF Viewport
                # Examples on using ByteImageProvider can be found by installing Isaac Sim
                # and searching for `set_bytes_data` under `~/.local/share/ov/pkg/isaac_sim-2023.1.1`.
                # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html
                # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ImageWithProvider.html
                self.ui_nerf_provider = ui.ByteImageProvider()
                # TODO: Potentially optimize with `set_bytes_data_from_gpu`
                self.ui_nerf_img = ui.ImageWithProvider(
                    self.ui_nerf_provider,
                    width=ui.Percent(100),
                    height=ui.Percent(100),
                )
                # TODO: Larger image size?
        self.update_ui()

    def update_ui(self):
        print("[omni.nerf.viewport] Updating UI")
        print(f"[omni.nerf.viewport] Selected Camera: {self.selected_camera_path}")
        # self.ui_lbl.text = f"Selected Camera: {self.selected_camera_path}"
        # Ref: https://forums.developer.nvidia.com/t/refresh-window-ui/221200
        self.ui_window.frame.rebuild()

    def on_btn_click(self):
        # TODO: Allow resetting the camera to a specific position
        # Below doesn't seem to work
        # stage: Usd.Stage = self.usd_context.get_stage()
        # prim: Usd.Prim = stage.GetPrimAtPath('/OmniverseKit_Persp')
        # # `UsdGeom.Xformable(prim).SetTranslateOp` doesn't seem to exist
        # prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0.1722))
        # prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(0, -152, 0))
        # print("translateOp", prim.GetAttribute("xformOp:translate").Get())
        # print("rotateXYZOp", prim.GetAttribute("xformOp:rotateXYZ").Get())
        print("[omni.nerf.viewport] (TODO) Reset Camera")

    def _get_selected_camera_path(self):
        """Get the selected camera prim. Return None if no camera is selected or the first selected prim isn't a camera."""
        # Ref: https://docs.omniverse.nvidia.com/workflows/latest/extensions/object_info.html#step-5-get-the-selected-prims-data
        selected_prim_paths = self.usd_context.get_selection().get_selected_prim_paths()
        if not selected_prim_paths:
            return None
        stage: Usd.Stage = self.usd_context.get_stage()
        selected_prim = stage.GetPrimAtPath(selected_prim_paths[0])
        assert type(selected_prim) == Usd.Prim
        if not selected_prim.IsA(UsdGeom.Camera):
            return None
        return selected_prim.GetPath()

    def _on_stage_event(self, event):
        """Called by stage_event_stream. We only care about selection changes."""
        print("[omni.nerf.viewport] on_stage_event", omni.usd.StageEventType(event.type))
        if event.type != int(omni.usd.StageEventType.SELECTION_CHANGED):
            return
        selected_camera_path = self._get_selected_camera_path()
        if self.selected_camera_path == selected_camera_path:
            # Skip if the selected camera hasn't changed
            print("[omni.nerf.viewport] Skip updating UI")
            return
        self.selected_camera_path = selected_camera_path
        self.update_ui()

    def _on_rendering_event(self, event):
        """Called by rendering_event_stream."""
        # No need to check event type, since there is only one event type: `NEW_FRAME`.
        if self.is_python_supported:
            viewport_api = get_active_viewport()
            # We chose to use Viewport instead of Isaac Sim's Camera Sensor to avoid dependency on Isaac Sim.
            # We want the extension to work with any Omniverse app, not just Isaac Sim.
            # Ref: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html
            camera_mat: Gf.Matrix4d = viewport_api.transform
            # TODO: Use viewport camera projection matrix `viewport_api.projection`?
            camera_position: Gf.Vec3d = camera_mat.ExtractTranslation()
            camera_rotation: Gf.Vec3d = camera_mat.ExtractRotation().Decompose(*Gf.Matrix3d())
            # Not same as below due to the potential difference in rotation matrix representation
            # ```
            # from scipy.spatial.transform import Rotation as R
            # camera_rotation: Gf.Vec3d = R.from_matrix(camera_mat.ExtractRotationMatrix()).as_euler('xyz', degrees=True) # in degrees
            # ```
            # TODO: Consider object transform (if it is moved or rotated)
            # No need to transform from Isaac Sim space to Nerfstudio space, since they are both in the same space.
            # Ref: https://github.com/j3soon/coordinate-system-conventions
            if camera_position != self.camera_position or camera_rotation != self.camera_rotation:
                self.camera_position = camera_position
                self.camera_rotation = camera_rotation
                print("[omni.nerf.viewport] New camera position:", camera_position)
                print("[omni.nerf.viewport] New camera rotation:", camera_rotation)
                self.rpyc_conn.execute(f'rq.update_camera({list(camera_position)}, {list(np.deg2rad(camera_rotation))})')
            image = self.rpyc_conn.eval('rq.get_rgb_image()')
            if image is not None:
                print("[omni.nerf.viewport] NeRF viewport updated")
                image = np.array(image) # received with shape (H*, W*, 3)
                image = cv2.resize(image, (self.rgba_w, self.rgba_h), interpolation=cv2.INTER_LINEAR) # resize to (H, W, 3)
                self.rgba[:,:,:3] = image * 255
        else:
            # If python version is not supported, render the dummy image.
            self.rgba[:,:,:3] = (self.rgba[:,:,:3] + np.ones((self.rgba_h, self.rgba_w, 3), dtype=np.uint8)) % 256
        self.ui_nerf_provider.set_bytes_data(self.rgba.flatten().tolist(), (self.rgba_w, self.rgba_h))

    def on_shutdown(self):
        print("[omni.nerf.viewport] omni nerf viewport shutdown")
        if self.is_python_supported:
            self.rpyc_conn.execute('del rq')

    def destroy(self):
        # Ref: https://docs.omniverse.nvidia.com/workflows/latest/extensions/object_info.html#step-3-4-use-usdcontext-to-listen-for-selection-changes
        self.stage_event_stream = None
        self.stage_event_delegate.unsubscribe()
