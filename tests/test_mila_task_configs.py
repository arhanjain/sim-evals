from __future__ import annotations

import importlib.util
import ast
import math
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "src" / "sim_evals" / "environments" / "mila" / "tasks.py"


def _load_tasks_module():
    module_name = "mila_tasks_config_test"
    spec = importlib.util.spec_from_file_location(module_name, TASKS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TASKS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MilaTaskConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_tasks_module()
        cls.tasks = cls.module.load_mila_tasks(ROOT / "configs" / "mila")

    def test_all_tasks_use_replacement_scene_and_seed_42(self):
        self.assertEqual(
            set(self.tasks),
            {
                "place_spoon_in_sink",
                "place_spoon_in_utensil_holder",
                "stack_red_bowl_into_grey_bowl",
            },
        )
        for task in self.tasks.values():
            self.assertEqual(task.scene_asset, "Task1/Scene.usd")
            self.assertEqual(task.seed, 42)
            self.assertEqual(task.success_hold_steps, 5)
            self.assertEqual(task.final_milestone, task.milestones[-1].name)

    def test_randomization_is_object_only_with_approved_ranges(self):
        sink = self.tasks["place_spoon_in_sink"]
        self.assertEqual(sink.object_spawn_bounds.x, (0.0, 0.0))
        self.assertEqual(sink.object_spawn_bounds.y, (-0.07, 0.07))
        self.assertEqual(sink.object_spawn_bounds.yaw, (-math.pi, math.pi))
        self.assertIsNone(sink.target_spawn_bounds)
        self.assertEqual(
            sink.object_region_points,
            ((0.0, -0.18, 0.0), (0.0, 0.18, 0.0)),
        )
        self.assertEqual(sink.target_region_prim_path, "Sink003/Sink003/Sites/reg_basin")

        holder = self.tasks["place_spoon_in_utensil_holder"]
        self.assertEqual(holder.object_spawn_bounds.x, (0.0, 0.0))
        self.assertEqual(holder.object_spawn_bounds.y, (-0.15, 0.0))
        self.assertEqual(holder.object_spawn_bounds.yaw, (-math.pi, math.pi))
        self.assertIsNone(holder.target_spawn_bounds)
        self.assertEqual(holder.spawn_constraints[0].kind, "segment_circle_clearance")

        bowls = self.tasks["stack_red_bowl_into_grey_bowl"]
        self.assertEqual(bowls.object_spawn_bounds.x, (0.0, 0.0))
        self.assertEqual(bowls.object_spawn_bounds.y, (0.0, 0.15))
        self.assertEqual(bowls.object_spawn_bounds.yaw, (0.0, 0.0))
        self.assertEqual(bowls.target_spawn_bounds.x, (-0.10, 0.10))
        self.assertEqual(bowls.target_spawn_bounds.y, (0.0, 0.166))
        self.assertEqual(bowls.target_spawn_bounds.yaw, (0.0, 0.0))
        self.assertEqual(bowls.spawn_constraints[0].kind, "minimum_planar_distance")

        for path in (ROOT / "configs" / "mila").glob("*.yaml"):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["randomization_scope"], "movable_task_objects_only")
            self.assertNotIn("robot", raw["randomization"])
            self.assertNotIn("cameras", raw["randomization"])

    def test_sphere_light_is_enabled_only_for_spoon_in_cup(self):
        enabled_tasks = {
            task_id
            for task_id, task in self.tasks.items()
            if task.sphere_light.enabled
        }
        self.assertEqual(enabled_tasks, {"place_spoon_in_utensil_holder"})

        environment_path = (
            ROOT
            / "src"
            / "sim_evals"
            / "environments"
            / "mila"
            / "mila_droid_environment.py"
        )
        tree = ast.parse(environment_path.read_text(encoding="utf-8"))
        scene_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MilaSceneCfg"
        )
        inherited_sphere_overrides = [
            node
            for node in scene_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "sphere_light"
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ]
        self.assertEqual(len(inherited_sphere_overrides), 1)

    def test_configs_define_exactly_three_robot_relative_cameras(self):
        for task in self.tasks.values():
            self.assertEqual(
                set(task.cameras),
                {"external_cam", "external_cam_2", "wrist_cam"},
            )
            self.assertTrue(task.cameras["wrist_cam"].inherited)

    def test_bowl_arm_is_raised_one_inch_without_moving_the_bowls(self):
        bowls = self.tasks["stack_red_bowl_into_grey_bowl"]
        self.assertAlmostEqual(bowls.robot_base_pos[2], 0.831519064555536)
        self.assertAlmostEqual(
            bowls.robot_base_pos[2] + bowls.object_init_pos_robot[2],
            0.9007219075693947,
        )
        self.assertAlmostEqual(
            bowls.robot_base_pos[2] + bowls.target_init_pos_robot[2],
            0.9111174353900667,
        )


if __name__ == "__main__":
    unittest.main()
