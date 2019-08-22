import numpy as np
import pandas as pd
import pygame
import scipy as sp
import socket
import traceback
from numpy import linalg as LA
from OpenGL import GL as gl
from OpenGL.GLU import gluPerspective
from pykalman import KalmanFilter


GRAVITATION = 9.81
PI = np.pi
RAD2DEG = 180. / PI
DEG2RAD = PI / 180.


def rotation_matrix(axis, theta):
    """Generate a rotation matrix to Rotate the matrix over the given axis by
    the given theta (angle)

    Uses the `Euler-Rodrigues
    <https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula>`_
    formula for fast rotations.

        :param axis: Three-dim. axis to rotate around of
        :type axis: np.array
        :param theta: Rotation angle in radians
    """

    axis = np.asarray(axis)
    # No need to rotate if there is no actual rotation
    if not axis.any():
        return np.zeros((3, 3))

    theta = 0.5 * np.asarray(theta)

    axis /= np.linalg.norm(axis)

    a = np.cos(theta)
    b, c, d = - axis * np.sin(theta)
    angles = a, b, c, d
    powers = [x * y for x in angles for y in angles]
    aa, ab, ac, ad = powers[0:4]
    ba, bb, bc, bd = powers[4:8]
    ca, cb, cc, cd = powers[8:12]
    da, db, dc, dd = powers[12:16]

    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def axis_angle(vector, axis_ref):
    """Returns axis-angle representation of a 3D vector.

        :param vector: Input vector
        :type vector: np.array
        :param axis_ref: Reference axis that fixes the plane of rotation
        :type axis_ref: np.array
        :return rot_axis: Rotation axis
        :return rot_angle: Rotation angle
        :rtype: np.array, float
    """
    rot_angle = np.arccos(np.dot(axis_ref, vector) /
                          LA.norm(axis_ref) / LA.norm(vector))
    if np.abs(rot_angle) <= 1.0e-6:
        rot_axis = np.array([1., 0., 0.])
    elif np.abs(rot_angle - np.pi) <= 1.0e-06:
        rot_axis = np.array([1., 0., 0.])
    else:
        rot_axis = -np.cross(axis_ref, vector)
        rot_axis /= np.linalg.norm(rot_axis)
    return rot_axis, rot_angle


def get_rotation_matrix(vector, axis_ref):
    """Returns the rotation matrix required for rotating the input vector to
    the reference axis.

        :param vector: Input vector
        :type vector: np.array
        :param axis_ref: Vector determining the target axis
        :type axis_ref: np.array
        :return: Rotation matrix
        :rtype: np.array
    """

    rot_axis, rot_angle = axis_angle(vector, axis_ref)

    return rotation_matrix(rot_axis, rot_angle)


def pca_axes(x):
    """Principal component analysis based on eigenvalue decomposition of
    covariance matrix. Based on one static input.
    Could be done dynamically, too.

        :param x: Input data
        :type x: np.array
        :return w: Principal components' weights
        :return axes: Principal axes as columns
        :rtype: np.array, np.array
    """
    x_ = np.mean(x, axis=0)
    cov = np.dot((x - x_).T, (x - x_)) / len(x)
    w, axes = LA.eig(cov)
    return w, axes


def pitch_roll_analysis(gravitation, axes):
    """
    Estimate pitch and roll angles from gravitation vector
    observations given the respective axes.
    :param gravitation: Observations of the gravitation vector.
    :type gravitation: np.array
    :param axes: Ship coordinate axes in sensor's xy-plane
    :type axes: np.array (2x2)
    :return: pitch: Estimated pitch angles in arrays.
    :return roll: Estimated roll angles
    :rtype: np.array, np.array
    """
    # rotate gravitation to xy-axes
    rot_xy = np.array([[axes[0, 0], axes[0, 1], 0.],
                       [axes[1, 0], axes[1, 1], 0.],
                       [0., 0., 1.]])
    g_xy = np.dot(rot_xy.T, gravitation.T).T

    # angles
    pitch = np.arctan2(g_xy[:, 1], g_xy[:, 2])
    roll = np.arctan2(-g_xy[:, 0],
                      np.sqrt(g_xy[:, 1] ** 2 + g_xy[:, 2] ** 2))
    return pitch, roll


def kinematics_sine_series(t, freq, amp):
    """Simulate kinematics of sinusoidal series form.

        :param t: Time
        :type t: np.array
        :param freq: Frequencies in the series
        :type freq: np.array
        :param amp: Amplitudes of series terms
        :type amp: np.array
        :return x: Position
        :return v: Velocity
        :return a: Acceleration
        :rtype: np.array, np.array, np.array
    """
    arg = np.outer(t, freq)
    x = np.dot(np.sin(arg), amp)
    v = np.dot(np.cos(arg), freq * amp)
    a = -1.0 * np.dot(np.sin(arg), freq ** 2 * amp)
    return x, v, a


def accelerometer_swing(lever, x_ang, v_ang, a_ang, a_lin):
    """Accelerometer on Swing. Calculate two-dimensional accelerometer reading
    from angular and linear acceleration components.

        :param lever: Position of the accelerometer at zero angle
        :type lever: np.array
        :param x_ang: Array of polar angles
        :type x_ang: np.array
        :param v_ang: Array of angular velocities
        :type v_ang: np.array
        :param a_ang: Array of angular accelerations
        :type a_ang: np.array
        :param a_lin: Array of additional translational accelerations
        :type a_lin: np.array
        :return: Total acceleration
        :rtype: np.array complex
    """
    z = lever[0] + 1j * lever[1]
    a_lin = a_lin[:, 0] + 1j * a_lin[:, 1]
    a_grav = -1j * GRAVITATION * np.exp(1j * x_ang)
    a_rot = -z * (1j * a_lin + v_ang ** 2)
    a_total = a_grav + a_rot + a_lin * np.exp(1j * x_ang)
    return a_total


def accelerometer_swing_sine(t, lever, freq_rot, amp_rot, freq_lin, amp_lin):
    """Simulate measurement of an accelerometer which is attached on a swing
    that oscillates according to a sinusoidal series. In addition to the
    signal caused by rotational movement, there is a manually tunable
    linear accelerometer component.

        :param t: Time
        :type t: np.array
        :param lever: Position of the accelerometer at zero angle
        :type lever: np.array
        :param freq_rot: Frequencies in the rotational sine series
        :type freq_rot: np.array
        :param amp_rot: Amplitudes in the rotational sine series
        :type amp_rot: np.array
        :param freq_lin: Frequencies in the translational sine series
        :type freq_lin: np.array
        :param amp_lin: Amplitudes in the translational sine series
        :type amp_lin:np. array
        :return: Simulated accelerometer measurement
        :rtype: np.array
    """
    # simulate sinusoidal kinematics
    x_ang, v_ang, a_ang = kinematics_sine_series(t, freq_rot, amp_rot)
    _, _, a_lin_vert = kinematics_sine_series(t, freq_lin, amp_lin)
    a_lin = np.array([[0., val] for val in a_lin_vert])  # only vertical

    # simulate accelerometer signal
    a_meas = accelerometer_swing(lever, x_ang, v_ang, a_ang, a_lin)
    return a_meas


class AhrSystem(object):
    """
    Attitude and heading reference system based on accelerometer and
    gyroscope measurement on an IMU. The algorithm is based on a standard
    Kalman filter formulation (S. Särkkä 2016). The observation model is
    the mere accelerometer reading. Evolution model is based on a differential
    equation that relates angular velocity and gravity.

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
        angular_velo_norm = LA.norm(angular_velo)
        theta = time_step * angular_velo_norm

        # matrix exponential using Rodrigues rotation formula
        evolution_matrix = np.identity(3) + \
                           np.sin(theta) * -cross_mat / angular_velo_norm + \
                           (1 - np.cos(theta)) * \
                           LA.matrix_power(cross_mat, 2) / \
                           angular_velo_norm ** 2
        return evolution_matrix

    def offline_estimate(self):
        """
        Offline attitude estimation method using Kalman-filtering
        :return:
        """

        # initialize Kalman-filter
        x_list = []
        t_list = []
        rot_list = []
        count = 0
        x = self.kf.initial_state_mean
        P = self.kf.initial_state_covariance
        first = True

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
                    rot_list.append(RAD2DEG * rot_ang * rot_ax)
                except ValueError:
                    print('Invalid data point: ', t, acc, gyr)

        self.gravitation_array = np.array(x_list)
        self.timestamp_array = np.array(t_list)
        self.rotation_array = np.array(rot_list)

    def online_estimate(self):
        """
        Online attitude estimation method using Kalman-filtering. Optional
        visualization via PyOpenGL. Sensor data transmission is carried out
        through UDP. Currently compatible with the Android operating system
        with suitable data transmission software (e.g. "Wireless IMU").
        :return:
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
                rot_ax /= LA.norm(rot_ax)
                rot_ang = RAD2DEG * np.arccos(x[2] / LA.norm(x))

                # update graphic
                if self.config['graphic']:
                    display = OnScreenDisplay.draw_orientation(rot_ang, rot_ax)

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                traceback.print_exc()     # prints a lot of stuff on the cml


class OrientationGraphic(object):
    """An object for visualization of imu orientation in global coordinates
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


def tilt_transition_matrix(state, dt):
    """Wrap this into inline transition function in Unscented Kalman filter.

        :param state: State variable for tilt model
        :type state: dict with keys 'x_ang', 'v_ang', 'freq_osc', 'lever' and
        float values
        :param dt: Time step
        :type dt: float
        :return: Transition matrix
        :rtype: np.array
    """
    n_state = sum(len(state[key]) for key in state)
    n_ang = state['x_ang'].size

    # form exponential matrix
    exponent = np.zeros((n_state, n_state))
    exponent[:n_ang, n_ang:2 * n_ang] = np.identity(n_ang)
    exponent[n_ang:2 * n_ang, :n_ang] = \
        -np.diag(state['freq_osc'] ** 2)
    transition_matrix = sp.linalg.expm(exponent * dt)

    return transition_matrix


def tilt_observation_function(state):
    """Wrap this into inline observation function in Unscented Kalman filter.

    Enables using a nonlinear term involving the angular frequency

        :param state: State variable for tilt model
        :type state: dict with keys 'x_ang', 'v_ang', 'freq_osc', 'lever' and
        float values
        :return: Observation vector
        :rtype: np.array
    """
    x_ang = state['x_ang']
    v_ang = state['v_ang']
    freq_osc = state['freq_osc']
    lever = state['lever']

    offset = np.array([0., GRAVITATION])
    lin0 = np.sum((lever[1] * freq_osc ** 2 - GRAVITATION) * x_ang) + \
           lever[0] * np.sum(v_ang) ** 2
    lin1 = np.sum(-lever[0] * freq_osc ** 2 * x_ang) + \
           lever[1] * np.sum(v_ang) ** 2
    return np.array([lin0, lin1]) + offset


def tilt_observation_transformation(state):
    """Build observation matrix and offset using the simplified observation
    model.

    The matrix itself is a non-linear function of the full state but
    if frequency and lever are fixed, then the model is affine.

        :param state: State variable for tilt model
        :type state: dict with keys 'x_ang', 'v_ang', 'freq_osc', 'lever' and
        float values
        :return:
    """
    n_state = sum(len(state[key]) for key in state)
    n_angle = state['angle'].size
    lever = state['lever']
    freq_osc = state['freq_osc']

    offset = np.array([0., GRAVITATION])

    mat0 = np.zeros((2, n_state))
    mat0[0, :n_angle] = lever[0] * freq_osc
    mat0[1, :n_angle] = -lever[1] * freq_osc
    mat1 = np.zeros((2, n_state))
    mat1[:, :n_angle] = GRAVITATION

    # form the full matrix
    mat = mat0 - mat1
    return mat, offset


if __name__ == '__main__':
    A = AhrSystem(path_to_data=None,
                  graphic=True,
                  display_resolution =(3*640, 2*480),
                  chunksize=4000,
                  qc=1.0e-02,
                  sigma=1.0e-01)
    A.online_estimate()

