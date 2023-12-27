import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length
from scipy.spatial.transform import Rotation as R


def camera_to_world_matrix(position, rotation):
    """
    Constructs a camera-to-world (c2w) transformation matrix,
    based on the position and rotation of the camera.

    Parameters
    ----------
    position : list[float]
        A 3-element list of floats representing the position of the camera.

    rotation : list[float]
        A 3-element list of floats representing the rotation of the camera, in euler angles.

    Returns
    ----------
    np.array
        A 4x4 camera-to-world matrix.
    """
    x, y, z = position
    world_position = [z, -x, y]

    # Obtain transformed rotations
    rotation_rad = np.array(rotation) * np.pi / 180
    v_c = np.array([0, 0, 1])
    R_uc = R.from_euler('xyz', rotation_rad).as_matrix()
    R_mu = np.array([[0, 0, 1,], [-1, 0, 0], [0, 1, 0]])
    R_mc = np.matmul(R_mu, R_uc)
    v_m = np.matmul(R_mc, v_c)
    unit_m = np.array([0, 0, -1])

    v1 = unit_m
    v2 = v_m
    cross_product = np.cross(v1, v2)

    yaw_angle = np.arctan2(cross_product[1], cross_product[0])
    pitch_angle = np.arcsin(cross_product[2])
    roll_angle = np.arctan2(-v1[2], v1[0])

    c2w_rotations = np.array([roll_angle, pitch_angle, yaw_angle])
    c2w_rotations_rad = R.from_euler('xyz', c2w_rotations).as_matrix()

    # Compose c2w matrix
    camera_to_world_matrix = np.eye(4)
    camera_to_world_matrix[:3, :3] = c2w_rotations_rad
    camera_to_world_matrix[:3, 3] = world_position

    return camera_to_world_matrix

def create_cameras(camera_to_world_matrix, width, height, fov):
    """
    Constructs a Cameras object based on a c2w matrix,
    and a camera configuration from RendererCameraConfig.

    Parameters
    ----------
    camera_to_world_matrix : np.array
        A 3-element list of floats representing the position of the camera.

    width : int
        The width of the camera.

    height : int
        The height of the camera.

    fov : float
        The vertical field-of-view of the camera.

    Returns
    ----------
    Cameras
        A Cameras object.
    """
    # Compute camera focal length
    focal_length = three_js_perspective_camera_focal_length(fov, height)

    # Only use the first 3 rows of the c2w matrix, as the last row is always [0 0 0 1].
    camera_to_worlds = torch.tensor(camera_to_world_matrix).view(4, 4)[:3].view(1, 3, 4)

    return Cameras(
        fx=torch.tensor([focal_length]),
        fy=torch.tensor([focal_length]),
        cx=width / 2,
        cy=height / 2,
        camera_to_worlds=camera_to_worlds,
        camera_type=CameraType.PERSPECTIVE,
        times=None,
    )
