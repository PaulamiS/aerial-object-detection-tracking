"""Constant-velocity Kalman filter for a 2D point. State: [x, y, vx, vy]."""
import numpy as np


class ConstantVelocityKF:
    def __init__(self, x, y, dt=1.0, q=0.01, r=5.0):
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)   # state estimate
        self.P = np.eye(4) * 100.0                          # state covariance
        self.F = np.array([[1, 0, dt, 0],                   # motion model
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],                    # we only measure x, y
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * q                              # process noise
        self.R = np.eye(2) * r                              # measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x                             # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)            # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()
