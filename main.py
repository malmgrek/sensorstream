import socket
from typing import List, Tuple

import numpy as np
from numpy import linalg
import pygame
from OpenGL import GL as gl
from OpenGL.GLU import gluPerspective
from pykalman import KalmanFilter


GRAVITATION = 9.81


# TODO: Scrap / refactor to submodules the simulation code
# TODO: Refactor the drawing code
# TODO: Is it possible to manage state better / async?
# TODO: Improve usability of the on-screen-display
# TODO: Test using some simulations


def normalize(v: np.ndarray) -> np.ndarray:
    return v / linalg.norm(v)


def build_euler_rodrigues(axis: np.ndarray, angle: float):
    """Rotation matrix to rotate a 3D vector over an axis by an angle

    Uses the `Euler-Rodrigues formula for fast rotations.

    Parameters
    ----------
    param axis : 3D axis to rotate about
    angle : Rotation angle in radians

    """

    # No need to rotate if there is no actual rotation
    if not axis.any():
        return np.zeros((3, 3))

    axis = normalize(axis)

    a = np.cos(0.5 * angle)
    b, c, d = - axis * np.sin(0.5 * angle)
    angles = a, b, c, d
    powers = [x * y for x in angles for y in angles]
    aa, ab, ac, ad = powers[0:4]
    ba, bb, bc, bd = powers[4:8]
    ca, cb, cc, cd = powers[8:12]
    da, db, dc, dd = powers[12:16]

    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])


def represent_axis_angle(v: np.ndarray, aref: np.ndarray) -> Tuple:
    """Axis angle representation of a 3D vector

    REVIEW Do we want to allow over 180 degree angles? Atm not allowed.

    Parameters
    ----------
    v : Rotated vector
    aref : Reference i.e. zero-angle axis

    """

    def calc_angle(a, b):
        return np.arccos(
            np.dot(normalize(a), normalize(b))
        )

    def calc_axis(a, b):
        return normalize(-np.cross(b, a))

    def is_small(x):
        return (np.abs(x) <= 1.0e-6) or (np.abs(x - np.pi) <= 1.0e-6)

    axis = calc_axis(v, aref)
    angle = calc_angle(v, aref)
    return (
        (np.array([1., 0., 0.]), angle) if is_small(angle)
        else (axis, angle)
    )


def evolution_matrix(gyro: List[float], dt: float) -> np.ndarray:
    """Assembles the evolution matrix for Kalman-filter

    Gravitation vector tracking

    Parameters
    ----------
    gyro : Angular velocity in radians
    dt : time step in seconds

    """
    # cross product operator with gyro
    A = np.array([
        [0.0, -gyro[2], gyro[1]],
        [gyro[2], 0.0, -gyro[0]],
        [-gyro[1], gyro[0], 0.0]
    ])
    I = np.identity(3)
    speed = linalg.norm(gyro)
    theta = dt * speed

    # matrix exponential using Rodrigues rotation formula
    return (
        I + np.sin(theta) * - A / speed +
        (1 - np.cos(theta)) * linalg.matrix_power(A, 2) / speed ** 2
    )


# FIXME As functions (online_estimate, offline_estimate)
class AhrSystem(object):
    """Attitude and heading reference system

    Based on accelerometer and gyroscope measurement on an IMU. The algorithm is
    based on a standard Kalman filter formulation (S. Särkkä 2016). The
    observation model is the mere accelerometer reading. Evolution model is
    based on a differential equation that relates angular velocity and gravity.

    :Example:

    >>> input_file = '../data/my-data-file.csv'
    >>> ahrs = imu16.AhrSystem(path_to_data=input_file)
    >>> ahrs.online_estimate()

    :param path_to_data: Relative path to the data file
    :type path_to_data: str
    :param fs: Sampling frequency
    :type fs: float
    :param chunksize: Size of data chunk
    :type chunksize: int
    :param graphic: Boolean for graphic
    :type graphic: bool
    :param display_resolution: Resolution of the on-screen display
    :type display_resolution: tuple
    :param qc: np.identity(3) * qc is transition covariance
    :type qc: float
    :param sigma: np.identity(3) * sigma ** 2 is observation covariance
    :type sigma: float
    :param dimensions: Labels for input data columns, currently hard-coded
    :type dimensions: list
    """

    DEFAULTS = {
        'path_to_data': None,
        'fs': 101.8,
        'chunksize': 2 ** 10,
        'graphic': False,
        'display_resolution': (640, 480),
        'qc': 0.1,
        'sigma': 1.0,
        'dimensions': ['time',
                       'acc_x', 'acc_y', 'acc_z',
                       'gyr_x', 'gyr_y', 'gyr_z']
    }

    def __init__(self, **kwargs):

        # update config
        config = {**self.DEFAULTS, **kwargs}

        # initialize KalmanFilter object
        R = config['sigma'] ** 2 * np.identity(3)
        Q = config['qc'] * np.identity(3)
        kf = KalmanFilter(transition_matrices=np.identity(3),
                          observation_matrices=np.identity(3),
                          transition_covariance=Q,
                          observation_covariance=R,
                          transition_offsets=np.zeros(3),
                          observation_offsets=np.zeros(3),
                          initial_state_mean=np.array([0.0, 0.0, 9.81]),
                          initial_state_covariance=np.identity(3)
                          )

        # construct data reader with default dimensions
        if config['path_to_data']:
            data_reader = pd.read_csv(
                config['path_to_data'],
                header=None,
                names=self.DEFAULTS['dimensions'],
                chunksize=config['chunksize'])
        else:
            data_reader = None

        # set attributes
        self.config = config
        self.data_reader = data_reader
        self.kf = kf

        # initialize other attributes
        self.gravitation_array = None
        self.timestamp_array = None
        self.rotation_array = None

    @classmethod
    def _evolution_matrix(cls, angular_velo, time_step):
        """
        Assembles the evolution matrix for the Kalman-filter that tracks
        gravitation vector.
        :param angular_velo: Angular velocity in rad.
        :param time_step: Time step in sec.
        :return: 3 x 3 evolution matrix as a numpy array.
        """
        # cross product operator with 'angular_velo'
        cross_mat = np.array([
            [0.0, -angular_velo[2], angular_velo[1]],
            [angular_velo[2], 0.0, -angular_velo[0]],
            [-angular_velo[1], angular_velo[0], 0.0]
        ])
        angular_velo_norm = linalg.norm(angular_velo)
        theta = time_step * angular_velo_norm

        # matrix exponential using Rodrigues rotation formula
        evolution_matrix = np.identity(3) + \
                           np.sin(theta) * -cross_mat / angular_velo_norm + \
                           (1 - np.cos(theta)) * \
                           linalg.matrix_power(cross_mat, 2) / \
                           angular_velo_norm ** 2
        return evolution_matrix

    def offline_estimate(self):
        """Offline attitude estimate

        """

        # initialize Kalman-filter
        x_list = []
        t_list = []
        rot_list = []
        count = 0
        x = self.kf.initial_state_mean
        P = self.kf.initial_state_covariance
        first = True

        # FIXME No no no data reader
        # FIXME Use reduce and not for loop
        for chunk in self.data_reader:
            # keep count of progress
            count += 1

            # timestamps and sensor readings
            t_val = chunk[['time']].values
            acc_val = chunk[['acc_x', 'acc_y', 'acc_z']].values
            gyr_val = chunk[['gyr_x', 'gyr_y', 'gyr_z']].values

            # run filter update loop
            for t, acc, gyr in zip(t_val, acc_val, gyr_val):
                try:
                    # current observation
                    y = acc

                    # initialize first time
                    if first:
                        t_prev = t - 0.1
                        first = False

                    # next filtered estimate
                    dt = t - t_prev
                    A = self._evolution_matrix(gyr, dt)
                    Q = self.config['qc'] * dt * np.identity(3)
                    next_filtered_state_mean, next_filtered_state_covariance = \
                        self.kf.filter_update(
                            filtered_state_mean=x,
                            filtered_state_covariance=P,
                            observation=y,
                            transition_matrix=A,
                            transition_covariance=Q
                        )

                    # update evolving filter variables
                    x = next_filtered_state_mean
                    P = next_filtered_state_covariance
                    t_prev = t

                    # update additional rotational variables
                    rot_ax, rot_ang = axis_angle(np.array([x[0], x[1], x[2]]),
                                                 np.array([0., 0., 1.]))

                    # append lists
                    x_list.append(x)
                    t_list.append(t)
                    rot_list.append(np.rad2deg(rot_ang) * rot_ax)
                except ValueError:
                    print('Invalid data point: ', t, acc, gyr)

        self.gravitation_array = np.array(x_list)
        self.timestamp_array = np.array(t_list)
        self.rotation_array = np.array(rot_list)

    def online_estimate(self):
        """Online attitude estimation method using Kalman-filtering

        Optional visualization via PyOpenGL. Sensor data transmission is carried
        out through UDP. Currently compatible with the Android operating system
        with suitable data transmission software (e.g. "Wireless IMU").

        """
        # initialize networking interface
        host = ''
        port = 5555
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind((host, port))

        # initialize window for graphical visualization
        if self.config['graphic']:
            OnScreenDisplay = OrientationGraphic(
                display_resolution=self.config['display_resolution'])

        # initialize loop for estimation
        x = self.kf.initial_state_mean
        P = self.kf.initial_state_covariance
        first = True
        display = True
        while display:
            try:
                # receive data from the socket
                msg, addr = s.recvfrom(8192)  # buffer size 8192
                msg_str = msg.decode('utf-8')
                msg_val = [float(val.strip()) for val in msg_str.split(',')]

                # sensor readings to array
                t = msg_val[0]
                acc = np.array(msg_val[2:5])
                gyr = np.array(msg_val[6:9])

                # latest observation
                y = acc

                # initialize previous time
                if first:
                    t_prev = t - 0.01
                    first = False

                # next filtered estimate
                dt = t - t_prev  # time-step
                A = self._evolution_matrix(gyr, dt)
                Q = self.config['qc'] * dt * np.identity(3)
                next_filtered_state_mean, next_filtered_state_covariance = \
                    self.kf.filter_update(
                        filtered_state_mean=x,
                        filtered_state_covariance=P,
                        observation=y,
                        transition_matrix=A,
                        transition_covariance=Q
                    )

                # update evolving variables
                x = next_filtered_state_mean
                P = next_filtered_state_covariance
                t_prev = t

                # rotational quantities
                rot_ax = np.cross(np.array([0.0, 0.0, 1.0]),
                                  np.array([x[0], -x[1], x[2]]))
                rot_ax /= linalg.norm(rot_ax)
                rot_ang = np.rad2deg(np.arccos(x[2] / linalg.norm(x)))

                # update graphic
                if self.config['graphic']:
                    display = OnScreenDisplay.draw_orientation(rot_ang, rot_ax)

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                traceback.print_exc()     # prints a lot of stuff on the cml


class Display():

    def __init__(resolution, video_flags):
        return


class OrientationGraphic(object):
    """Visualization of IMU orientation in global coordinates

    """

    DEFAULTS = {
        'display_resolution': (640, 480),
        'video_flags': pygame.OPENGL | pygame.DOUBLEBUF
    }

    def __init__(self, **kwargs):

        # update configuration parameters
        assert set(self.DEFAULTS) >= set(kwargs), \
            'Unknown keyword arguments as parameters'
        self.params = self.DEFAULTS.copy()
        self.params.update(kwargs)

        # define attributes
        pygame.init()
        screen = pygame.display.set_mode(self.params['display_resolution'],
                                         self.params['video_flags'])
        pygame.display.set_caption("Press Esc to quit")

        # display perspective
        self._resize()

        # colors and depth buffer
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClearDepth(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)

    def _resize(self):
        """Resize screen and define perspective

        """
        gl.glViewport(0,
                      0,
                      self.params['display_resolution'][0],
                      self.params['display_resolution'][1])
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gluPerspective(45,
                       self.params['display_resolution'][0] /
                       self.params['display_resolution'][1],
                       0.1,
                       100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def _draw_text(self, position, text_string):
        """Draw text on the on-screen display.

        """

        font = pygame.font.SysFont("Courier", 18, True)
        text_surface = font.render(text_string, True, (255, 255, 255, 255),
                                   (0, 0, 0, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        gl.glRasterPos3d(*position)
        gl.glDrawPixels(text_surface.get_width(),
                        text_surface.get_height(),
                        gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        text_data)

    def _draw_cuboid(self):
        """Draw a hexahedron on the on-screen display.

        """
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

    def draw_orientation(self, rotation_angle, rotation_axis):
        """Update the current orientation image

            :param rotation_angle: Angle of rotation in degrees
            :type rotation_angle: float
            :param rotation_axis: Axis of rotation
            :type rotation_axis: np.array
            :return: Boolean of exit message reception
            :rtype: bool
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        gl.glTranslatef(0.0, 0.0, -7.0)

        # draw some text to the on-screen display
        osd_text = 'Inclination: ' + '{:.3f}'.format(rotation_angle) + ' deg.' \
                   + ", Axis: " + np.array_str(rotation_axis, precision=3)
        self._draw_text((-2.0, 1.5, 2.5), osd_text)

        # rotate the image
        gl.glRotatef(rotation_angle,
                     rotation_axis[0],
                     rotation_axis[2],
                     rotation_axis[1])

        # draw the geometrical shape and update screen
        self._draw_cuboid()
        pygame.display.flip()

        # interactions for quitting
        event = pygame.event.poll()
        if event.type == pygame.QUIT or \
                (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            return False
        else:
            return True


if __name__ == '__main__':
    A = AhrSystem(path_to_data=None,
                  graphic=True,
                  display_resolution =(3*640, 2*480),
                  chunksize=4000,
                  qc=1.0e-02,
                  sigma=1.0e-01)
    A.online_estimate()

