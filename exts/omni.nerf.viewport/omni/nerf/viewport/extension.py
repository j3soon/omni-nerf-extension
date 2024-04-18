import numpy as np
import omni.ext
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Usd, UsdGeom


# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    print("[omni.nerf.viewport] some_public_function was called with x: ", x)
    return x ** x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class OmniNerfViewportExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        # To see the Python print output, open the `Script Editor`.
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
        # Build UI
        self.build_ui()

    def build_ui(self):
        """Build the UI. Should be called upon startup."""
        # Please refer to the `Omni::UI Doc` tab in Omniverse Code for efficient development.
        # Ref: https://youtu.be/j1Pwi1KRkhk
        # Ref: https://github.com/NVIDIA-Omniverse
        # Ref: https://youtu.be/dNLFpVhBrGs
        self.ui_window = ui.Window("NeRF Viewport")

        with self.ui_window.frame:
            with ui.VStack(height=0):
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
                self.viewport_api = get_active_viewport()
                # We chose to use Viewport instead of Isaac Sim's Camera Sensor to avoid dependency on Isaac Sim.
                # We want the extension to work with any Omniverse app, not just Isaac Sim.
                # Ref: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html

                # NeRF Viewport
                # Examples on using ByteImageProvider can be found by installing Isaac Sim
                # and searching for `set_bytes_data` under `~/.local/share/ov/pkg/isaac_sim-2023.1.1`.
                # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ByteImageProvider.html
                # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/omni.ui/omni.ui.ImageWithProvider.html
                self.ui_nerf_provider = ui.ByteImageProvider()
                # TODO: Potentially optimize with `set_bytes_data_from_gpu`
                w, h = 256, 256
                self.ui_nerf_img = ui.ImageWithProvider(
                    self.ui_nerf_provider,
                    width=w,
                    height=h,
                )
                # TODO: Larger image size?
                # TODO: Get viewport data and show it
                # Ref: https://forums.developer.nvidia.com/t/how-can-i-grab-the-viewport-or-the-camera-rendering-in-a-python-script/238365/2
                # TODO: Get viewport matrices and show it
                print("Viewport Projection", self.viewport_api.projection)
                print("Viewport Transform", self.viewport_api.transform)

                rgba = np.ones((w, h, 4), dtype=np.uint8) * 128
                rgba[:,:,3] = 255
                self.ui_nerf_provider.set_bytes_data(rgba.flatten().tolist(), (w, h))
                # TODO: Update as the viewport moves
                # Currently, the ByteImageProvider is only updated upon building the UI,
                # which means the image is not updated when the viewport moves.
                self.ui_lbl = ui.Label("(To Be Updated)")
        self.update_ui()

    def update_ui(self):
        print("[omni.nerf.viewport] Updating UI")
        print(f"[omni.nerf.viewport] Selected Camera: {self.selected_camera_path}")
        self.ui_lbl.text = f"Selected Camera: {self.selected_camera_path}"
        # Ref: https://forums.developer.nvidia.com/t/refresh-window-ui/221200
        self.ui_window.frame.rebuild()

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
        print("[omni.nerf.viewport] on_rendering_event", omni.usd.StageRenderingEventType(event.type))
        # No need to check event type, since there is only one event type: `NEW_FRAME`.

    def on_shutdown(self):
        print("[omni.nerf.viewport] omni nerf viewport shutdown")

    def destroy(self):
        # Ref: https://docs.omniverse.nvidia.com/workflows/latest/extensions/object_info.html#step-3-4-use-usdcontext-to-listen-for-selection-changes
        self.stage_event_stream = None
        self.stage_event_delegate.unsubscribe()
