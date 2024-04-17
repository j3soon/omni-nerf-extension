import omni.ext
import omni.ui as ui
import omni.usd
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
        # Listen to selection changes
        # Ref: https://docs.omniverse.nvidia.com/workflows/latest/extensions/object_info.html#step-3-4-use-usdcontext-to-listen-for-selection-changes
        self.events = self.usd_context.get_stage_event_stream()
        self.stage_event_delegate = self.events.create_subscription_to_pop(
            self._on_stage_event, name="Object Info Selection Update"
        )
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

    def on_shutdown(self):
        print("[omni.nerf.viewport] omni nerf viewport shutdown")

    def destroy(self):
        # Ref: https://docs.omniverse.nvidia.com/workflows/latest/extensions/object_info.html#step-3-4-use-usdcontext-to-listen-for-selection-changes
        self.events = None
        self.stage_event_delegate.unsubscribe()
