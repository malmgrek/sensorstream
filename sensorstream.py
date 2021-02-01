"""Visualize sensor data streams in Python"""

import argparse
import json
import logging
import socket
from time import sleep
from typing import Callable, List

import numpy as np
from OpenGL import GL as gl
from OpenGL.GLU import gluPerspective
import pygame
from pykalman import KalmanFilter


def evolution_matrix(gyro: List[float], dt: float) -> np.ndarray:
    """Assembles the evolution matrix for Kalman-filter

    Based on the following article:

        S. Särkkä et. al, Adaptive Kalman filtering and smoothing for
        gravitation tracking in mobile systems

    """
    # cross product operator with gyro
    A = np.array([
        [0.0, -gyro[2], gyro[1]],
        [gyro[2], 0.0, -gyro[0]],
        [-gyro[1], gyro[0], 0.0]
    ])
    I = np.identity(3)
    speed = np.linalg.norm(gyro)
    theta = dt * speed

    # matrix exponential using Rodrigues rotation formula
    return (
        I + np.sin(theta) * - A / speed +
        (1 - np.cos(theta)) * np.linalg.matrix_power(A, 2) / speed ** 2
    )


def init_opengl(resolution):
    """Initialize so that OpenGL drawings draw properly

    """

    # Resize
    gl.glViewport(0, 0, resolution[0], resolution[1])
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gluPerspective(45, resolution[0] / resolution[1], 0.1, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    # Colors and depth buffer
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClearDepth(1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)


def draw_cuboid(angle: float, axis: List):
    """Draw OpenGL cuboid

    """

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()
    gl.glTranslatef(0.0, 0.0, -7.0)

    # # Write some text
    font = pygame.font.SysFont("Courier", 18, True)
    text = (
        "Inclination: {:.2f} deg., Axis: [{:.2f}, {:.2f}, {:.2f}]".format(
            angle, *axis
        )
    )
    text_surf = font.render(text, True, (255, 255, 255, 255), (0, 0, 0, 255))
    text_data = pygame.image.tostring(text_surf, "RGBA", True)
    gl.glRasterPos3d(-2, 1.5, 2.5)
    gl.glDrawPixels(
        text_surf.get_width(),
        text_surf.get_height(),
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        text_data
    )

    # NOTE: y and z swapped places
    gl.glRotatef(angle, axis[0], axis[2], axis[1])

    gl.glBegin(gl.GL_QUADS)

    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glVertex3f(1.0, 0.2, -1.0)
    gl.glVertex3f(-1.0, 0.2, -1.0)
    gl.glVertex3f(-1.0, 0.2, 1.0)
    gl.glVertex3f(1.0, 0.2, 1.0)

    gl.glColor3f(1.0, 0.5, 0.0)
    gl.glVertex3f(1.0, -0.2, 1.0)
    gl.glVertex3f(-1.0, -0.2, 1.0)
    gl.glVertex3f(-1.0, -0.2, -1.0)
    gl.glVertex3f(1.0, -0.2, -1.0)

    gl.glColor3f(1.0, 0.0, 0.0)
    gl.glVertex3f(1.0, 0.2, 1.0)
    gl.glVertex3f(-1.0, 0.2, 1.0)
    gl.glVertex3f(-1.0, -0.2, 1.0)
    gl.glVertex3f(1.0, -0.2, 1.0)

    gl.glColor3f(1.0, 1.0, 0.0)
    gl.glVertex3f(1.0, -0.2, -1.0)
    gl.glVertex3f(-1.0, -0.2, -1.0)
    gl.glVertex3f(-1.0, 0.2, -1.0)
    gl.glVertex3f(1.0, 0.2, -1.0)

    gl.glColor3f(0.0, 0.0, 1.0)
    gl.glVertex3f(-1.0, 0.2, 1.0)
    gl.glVertex3f(-1.0, 0.2, -1.0)
    gl.glVertex3f(-1.0, -0.2, -1.0)
    gl.glVertex3f(-1.0, -0.2, 1.0)

    gl.glColor3f(1.0, 0.0, 1.0)
    gl.glVertex3f(1.0, 0.2, -1.0)
    gl.glVertex3f(1.0, 0.2, 1.0)
    gl.glVertex3f(1.0, -0.2, 1.0)
    gl.glVertex3f(1.0, -0.2, -1.0)

    gl.glEnd()


def create_socket_processor(
        # Similar to the function in reduce, can be
        # e.g. Kalman filter
        reduct: Callable,
        init: Callable,
        display: bool=False
):
    """Create different streaming data processing methods

    Optional support for on-screen visualization.

    FIXME: If server terminates process, window will hang.

    """

    def run(
            host: str,
            port: int,
            buffer: int,
            terminate: Callable=lambda x: False,
            resolution=(640, 480),
            video_flags=(pygame.OPENGL | pygame.DOUBLEBUF),
            **kwargs
    ):

        if display:
            pygame.init()
            pygame.display.set_mode(resolution, video_flags)
            pygame.display.set_caption("Press Esc to quit")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            # If you're not enabling the display, then init should ignore
            # those arguments
            cum = init(resolution=resolution, **kwargs)
            while not terminate(cum):
                if display:
                    event = pygame.event.poll()
                    if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and
                            event.key == pygame.K_ESCAPE
                    ):
                        pygame.quit()
                        return cum
                msg = s.recv(buffer)
                cum = reduct(cum, msg)
        return cum

    return run


def simple(*args, **kwargs):
    """Print readings on standard I/O

    Example
    -------

    .. code-block::

        simple(host="123.456.78.90", port=1234, buffer=8192)

    """
    return create_socket_processor(
        reduct=lambda cum, msg: print(msg.decode().split("\n")[-2]),
        init=lambda **kwargs: None
    )(*args, **kwargs)


def naive(*args, **kwargs):
    """Naive gravitation estimation

    Accelerometer-only gravitation vector online estimation

    """

    def reduct(cum, msg):
        try:
            y = np.array(
                json.loads(msg.decode().split("\n")[-2])
                .get("accelerometer")
                .get("value")
            )
        except json.JSONDecodeError as err:
            logging.warning(err)
            return (None, None)
        axis = np.cross(
            [0, 0, 1],
            [y[0], -y[1], y[2]]
        )
        axis = axis / np.linalg.norm(axis)
        angle = np.rad2deg(np.arccos(y[2] / np.linalg.norm(y)))
        sleep(0.05)
        draw_cuboid(angle, axis)
        pygame.display.flip()
        return (angle, axis)

    def init(resolution, **kwargs):
        init_opengl(resolution)
        return (None, None)

    return create_socket_processor(
        reduct=reduct,
        init=init,
        display=True
    )(*args, **kwargs)


def kalman(qc=0.1, sigma=1.0, *args, **kwargs):
    """Quaternion-free Kalman filter gravitation estimation

    """

    dt = 0.05
    I = np.identity(3)

    def reduct(cum, msg):
        (x, P, kalman_filter) = cum
        try:
            y = json.loads(msg.decode().split("\n")[-2])
            acc = y.get("accelerometer").get("value")
            gyro = y.get("gyroscope").get("value")
        except json.JSONDecodeError as err:
            logging.warning(err)
            return (x, P, kalman_filter)
        A = evolution_matrix(gyro, dt)
        Q = qc * dt * I
        (x, P) = kalman_filter.filter_update(
            filtered_state_mean=x,
            filtered_state_covariance=P,
            observation=acc,
            transition_matrix=A,
            transition_covariance=Q
        )
        axis = np.cross([0, 0, 1], [x[0], -x[1], x[2]])
        axis = axis / np.linalg.norm(axis)
        angle = np.rad2deg(np.arccos(x[2] / np.linalg.norm(x)))
        sleep(dt)
        draw_cuboid(angle, axis)
        pygame.display.flip()
        return (x, P, kalman_filter)

    def init(resolution):
        init_opengl(resolution)
        x = np.array([0.0, 0.0, 9.81])
        P = I
        kalman_filter = KalmanFilter(
            transition_matrices=I,
            observation_matrices=I,
            transition_covariance=qc*I,
            observation_covariance=sigma*I,
            transition_offsets=np.zeros(3),
            observation_offsets=np.zeros(3),
            initial_state_mean=x,
            initial_state_covariance=P
        )
        return (x, P, kalman_filter)

    return create_socket_processor(
        reduct=reduct,
        init=init,
        display=True
    )(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foobar")
    parser.add_argument(
        "--host",
        type=str,
        help="Host address, e.g. 123.456.78.90"
    )
    parser.add_argument(
        "--port", type=int, help="Port number, e.g. 3400"
    )
    parser.add_argument(
        "--buffer",
        type=int,
        help="Received buffer size in bits.",
        default=8192
    )
    parser.add_argument(
        "--method",
        type=str,
        help=(
            "Data stream processing method. " +
            "Options: simple, naive, kalman"
        )
    )
    args = parser.parse_args()
    host = args.host
    port = args.port
    buffer = args.buffer
    method = args.method
    if method == "simple":
        simple(host=host, port=port, buffer=buffer)
    if method == "naive":
        naive(host=host, port=port, buffer=buffer)
    if method == "kalman":
        kalman(host=host, port=port, buffer=buffer)
