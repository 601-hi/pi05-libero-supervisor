"""Reset one LIBERO task and read deployable-equivalent kinematic metrics."""

import json
import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from kinematic_safety import from_robosuite_robot


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    task_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_file, camera_heights=64, camera_widths=64)
    try:
        env.seed(7)
        env.reset()
        env.set_init_state(suite.get_task_init_states(0)[0])
        robot = env.env.robots[0]
        metrics = from_robosuite_robot(robot)
        payload = {
            "task": task.language,
            "jacobian_shape": list(robot.controller.J_full.shape),
            "joint_count": int(robot._joint_positions.size),
            "metrics": metrics.to_dict(),
            "deployability_note": (
                "Inputs correspond to a robot kinematic model plus encoder joint positions; "
                "no object pose, contact truth, reward, or task success is used."
            ),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
