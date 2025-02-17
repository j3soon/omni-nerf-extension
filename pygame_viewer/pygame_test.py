import argparse
import time

import cv2
import numpy as np
import pygame
import rpyc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument("--rpyc", type=bool, default=False)
    args = parser.parse_args()
    return args

def main(args):
    if not args.rpyc:
        # Remote: Make Connection & Import
        conn = rpyc.classic.connect(args.host, args.port)
        conn.execute('from nerfstudio_renderer import NerfStudioRenderQueue')
        conn.execute('from pathlib import Path')
        conn.execute('import torch')
        conn.execute('import numpy as np')
    else:
        from nerfstudio_renderer import NerfStudioRenderQueue
        from pathlib import Path
        import torch

    if not args.rpyc:
        # Create a Remote NerfStudioRenderQueue
        conn.execute(f'rq = NerfStudioRenderQueue(model_config_path=Path("{args.model_config_path}"), checkpoint_path="{args.model_checkpoint_path}", device=torch.device("{args.device}"))')
    else:
        rq = NerfStudioRenderQueue(
            model_config_path=Path(args.model_config_path),
            checkpoint_path=args.model_checkpoint_path,
            device=torch.device(args.device),
        )

    # Initialize Pygame
    pygame.init()

    # Set the width and height of the window
    width, height = 640, 360
    window_size = (width, height)

    # Create a Pygame window
    screen = pygame.display.set_mode(window_size)

    # Create a clock to control the frame rate
    clock = pygame.time.Clock()

    # Camera curve time & global screen buffer
    camera_curve_time = 0
    screen_buffer = np.zeros((width, height, 3), dtype=np.uint8)

    # Camera pose for the poster NeRF model
    camera_position = [0, 0, 0]
    camera_rotation = [0, 0, 0]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Retrieve image
        if not args.rpyc:
            image = conn.eval('rq.get_rgb_image()')
        else:
            image = rq.get_rgb_image()
        if image is not None:
            image = np.array(image) # received with shape (H*, W*, 3)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) # resize to (H, W, 3)
            image = np.transpose(image, (1, 0, 2))
            screen_buffer[:] = image * 255

        animation_progress = (np.sin(camera_curve_time) + 1) / 2

        # Cover the screen buffer with an indicator of camera position
        hud_width, hud_height = 100, 50
        bar_x, bar_y = 20, 24
        bar_w, bar_h = 60, 2
        # white background
        camera_position_indicator = np.ones((hud_width, hud_height, 3)) * 255
        # horizontal line
        camera_position_indicator[bar_x:bar_x+bar_w, bar_y:bar_y+bar_h, :] = 0
        # square indicator of current position
        hud_x = round(bar_x + bar_w * animation_progress)
        camera_position_indicator[hud_x-5:hud_x+5, 20:30, :] = 0
        screen_buffer[width-hud_width:, height-hud_height:, :] = camera_position_indicator

        # Convert the NumPy array to a Pygame surface
        image_surface = pygame.surfarray.make_surface(screen_buffer)

        # Blit the surface to the screen
        screen.blit(image_surface, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        clock.tick(30)

        # Move Camera
        camera_position[2] = animation_progress

        # Update Camera
        if not args.rpyc:
            conn.execute(f'rq.update_camera({camera_position}, {camera_rotation})')
        else:
            rq.update_camera(camera_position, camera_rotation)

        if int(time.time()) % 5 == 0:
            camera_curve_time += 1.0 / 30.0

    if not args.rpyc:
        # Delete remote render queue
        conn.execute('del rq')

    # Quit Pygame
    pygame.quit()

if __name__ == '__main__':
    main(parse_args())
