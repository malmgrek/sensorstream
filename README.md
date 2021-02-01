# Visualize mobile phone sensor data with Python

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Summary](#summary)
- [Introduction](#introduction)
- [Instructions](#instructions)
    - [Requirements](#requirements)
        - [Client side (laptop computer)](#client-side-laptop-computer)
        - [Server side (mobile phone)](#server-side-mobile-phone)
- [References](#references)

<!-- markdown-toc end -->


## Summary
Stream and visualize mobile phone sensor data in Python. As an example
algorithm, a Quaternion-free inclination tracking Kalman filter.

## Introduction

Mobile phones contain many interesting sensors that can be useful tools when
developing e.g. sensor fusion algorithms for drones. Python is useful for
quickly experimenting with sensor data processing algorithms before implementing
the embedded version. This project contains an example implementation of a
pipeline consisting of

- Data receiving from mobile phone,
- Sensor fusion algorithm for estimating sensor orientation,
- Real-time visualization on PC.

The inclination (gravitation) tracking in based on a quaternion-free method for
estimating gravitation direction in sensor's coordinates. Visualization using
OpenGL and Pygame.

## Instructions

### Requirements

#### Client side (laptop computer)

Python programming environment with the following additional packages

- PyGame: `pip install -U pygame`
- PyOpenGL: `pip install -U PyOpenGL`
- Numpy: `pip install numpy`
- PyKalman: `pip install pykalman`

#### Server side (mobile phone)

Currently tested only with the Android app
[SensorStreamer](https://github.com/yaqwsx/SensorStreamer "SensorStreamer") that sets up a mobile server for sending sensor data.

1. Install the app
2. Configure a data package with gyroscope and accelerometer data
3. Configure a connection with your favorite port (e.g. 3400)
4. Find out our mobile phone IP address (Search "IP Address")
5. Start a stream in the app with `Lowest possible period`
6. Run `python sensorstreamer.py --host=123.456.78.90 --port=1234 --buffer=8192
   --method=naive`
   
## References

S. Särkkä et. al, [_Adaptive Kalman filtering and smoothing for gravitation tracking in mobile systems_](https://ieeexplore.ieee.org/abstract/document/7346762)

