import math
import random
import pygame
import numpy as np
import cv2
import argparse
import time
import rpyc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=7007)
    parser.add_argument("--model_config_path", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    # Remote: Make Connection & Import
    conn = rpyc.classic.connect(args.host, args.port)
    conn.execute('from nerfstudio_renderer import NerfStudioRenderQueue')
    conn.execute('from pathlib import Path')

    # Create a Remote NerfStudioRenderQueue
    conn.execute(f'rq = NerfStudioRenderQueue(model_config_path=Path("{args.model_config_path}"))')

    # Initialize Pygame
    pygame.init()

    # Set the width and height of the window
    width, height = 640, 480
    window_size = (width, height)

    # Create a Pygame window
    screen = pygame.display.set_mode(window_size)

    # Create a clock to control the frame rate
    clock = pygame.time.Clock()

    # Camera curve time & global screen buffer
    camera_curve_time = 0
    screen_buffer = np.zeros((width, height, 3), dtype=np.uint8)

    # Camera pose
    camera_position = [0, 0, 0.1722]
    camera_rotation = [0, -152, 0]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Retrieve image
        image = conn.eval('rq.get_rgb_image()')
        if image is not None:
            image = np.array(image)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)
            screen_buffer = image * 255

        # Cover the screen buffer with an indicator of camera position
        camera_position_indicator = np.ones((100, 50, 3)) * 255
        camera_position_indicator[20:80, 24:26, :] = 0
        camera_position_on_map = round(camera_position[0] * 60 + 20)
        camera_position_indicator[camera_position_on_map-5:camera_position_on_map+5, 20:30, :] = 0
        screen_buffer[width-100:, height-50:, :] = camera_position_indicator

        # Convert the NumPy array to a Pygame surface
        image_surface = pygame.surfarray.make_surface(screen_buffer)

        # Blit the surface to the screen
        screen.blit(image_surface, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        clock.tick(30)

        # Move Camera
        camera_position[0] = (np.sin(camera_curve_time) + 1) / 2

        # Update Camera
        conn.execute(f'rq.update_camera({camera_position}, {camera_rotation})')
        
        if int(time.time()) % 2 == 0:
            camera_curve_time += 1.0 / 30.0

    # Delete remote render queue
    conn.execute('del rq')

    # Quit Pygame
    pygame.quit()

if __name__ == '__main__':
    main(parse_args())
