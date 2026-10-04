"""SDK odometry covariance, kept separate from SLAM and calibrated error bounds.

cuVSLAM b405f13 exposes row-major xyz/fixed-axis XYZ covariance in the
odometry world frame. It does not expose a covariance for the SLAM pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sensing.models import vector, wire
from .frames import TransformSample


# Numerical checks for the SDK's float32 output, not physical tolerances.
SYMMETRY_TOLERANCE = 32 * float(np.finfo(np.float32).eps)
PSD_TOLERANCE = 6 * SYMMETRY_TOLERANCE


@dataclass(frozen=True)
class OdometryDiagnostic:
    pose: TransformSample
    covariance_xyz_rpy: tuple | None

    def __post_init__(self):
        if (not isinstance(self.pose, TransformSample) or self.pose.static
                or self.pose.stamp.measurement_kind != 'estimated'
                or self.pose.position_error_m is not None
                or self.pose.angular_error_rad is not None):
            raise ValueError('odometry diagnostic requires an estimated pose with unknown error bounds')
        if self.covariance_xyz_rpy is None:
            return
        values = vector(self.covariance_xyz_rpy, 36, 'odometry covariance')
        matrix = np.asarray(values, dtype=np.float64).reshape(6, 6)
        diagonal = np.diag(matrix)
        if np.any(diagonal < 0):
            raise ValueError('odometry covariance has a negative variance')
        zero = diagonal == 0
        if np.any(matrix[zero, :] != 0) or np.any(matrix[:, zero] != 0):
            raise ValueError('zero odometry variance has a nonzero cross covariance')
        # Congruence removes the units/scales from the numeric checks. Original
        # values remain untouched, including singular and zero covariance.
        scale = np.sqrt(np.where(zero, 1., diagonal))
        with np.errstate(over='ignore', invalid='ignore'):
            normalized = matrix / scale[:, None] / scale[None, :]
        if (not np.isfinite(normalized).all()
                or not np.allclose(normalized, normalized.T, rtol=0., atol=SYMMETRY_TOLERANCE)):
            raise ValueError('odometry covariance is not symmetric')
        eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) * .5)
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] < -PSD_TOLERANCE:
            raise ValueError('odometry covariance is not positive semidefinite')
        object.__setattr__(self, 'covariance_xyz_rpy', values)

    def as_dict(self):
        return {
            'pose': wire(self.pose),
            'covariance_xyz_rpy': self.covariance_xyz_rpy,
            'covariance_status': 'unknown' if self.covariance_xyz_rpy is None else 'reported',
            'covariance_frame_id': self.pose.parent,
            'covariance_layout': 'row-major 6x6',
            'variable_order': ('x', 'y', 'z', 'rotation_x', 'rotation_y', 'rotation_z'),
            'variable_units': ('m', 'm', 'm', 'rad', 'rad', 'rad'),
            'rotation_convention': 'fixed-axis XYZ',
            'calibrated_error_bound': False,
            'physical_admission': False,
        }
