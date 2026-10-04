import numpy as np

from vla_supervisor.kinematic_safety import measure_public_kinematic_safety


class Data:
    qpos = np.array([0.0, 0.5])

    @staticmethod
    def get_site_jacp(_name):
        return np.array([[1, 0], [0, 2], [0, 0.5]])


class Model:
    jnt_range = np.array([[-1, 1], [-2, 2]], dtype=float)
    jnt_limited = np.array([1, 1], dtype=int)


class Sim:
    data = Data()
    model = Model()


class Controller:
    eef_name = "grip"
    qvel_index = [0, 1]


class Robot:
    _ref_joint_pos_indexes = [0, 1]
    _ref_joint_indexes = [0, 1]
    controller = Controller()


def test_public_kinematic_measurement_reports_limits_and_jacobian_sigma():
    result = measure_public_kinematic_safety(Sim(), Robot())
    assert result["kinematic_measurement_status"] == "ok"
    assert result["joint_margin"] == 1.0
    assert result["singularity_sigma"] > 0


def test_measurement_failure_is_explicitly_unavailable():
    result = measure_public_kinematic_safety(object(), object())
    assert result["joint_margin"] is None
    assert result["singularity_sigma"] is None
    assert result["kinematic_measurement_status"].startswith("unavailable:")

