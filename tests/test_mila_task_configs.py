from __future__ import annotations

import importlib.util
import ast
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "src" / "sim_evals" / "environments" / "mila" / "tasks.py"
GENERATOR_PATH = ROOT / "scripts" / "generate_mila_initial_conditions.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MilaTaskConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module("mila_tasks_config_test", TASKS_PATH)
        cls.generator = _load_module("mila_initial_conditions_test", GENERATOR_PATH)
        cls.tasks = cls.module.MILA_TASKS

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

    def test_generation_recipe_preserves_approved_object_only_ranges(self):
        sink = self.generator.TASK_SPECS["place_spoon_in_sink"]
        spoon = sink.actors["spoon"]
        self.assertEqual(spoon.bounds.x, (0.0, 0.0))
        self.assertEqual(spoon.bounds.y, (-0.07, 0.07))
        self.assertEqual(spoon.bounds.yaw, (-math.pi, math.pi))
        self.assertEqual(sink.constraints[0].kind, "segment_aabb_clearance")
        self.assertEqual(sink.constraints[0].asset_b, "sink")
        self.assertAlmostEqual(sink.constraints[0].clearance, 0.015)

        holder = self.generator.TASK_SPECS["place_spoon_in_utensil_holder"]
        spoon = holder.actors["spoon"]
        self.assertEqual(spoon.bounds.x, (0.0, 0.0))
        self.assertEqual(spoon.bounds.y, (-0.15, 0.0))
        self.assertEqual(spoon.bounds.yaw, (-math.pi, math.pi))
        self.assertIsNone(holder.actors["utensil_holder"].bounds)
        self.assertEqual(holder.constraints[0].kind, "segment_circle_clearance")

        bowls = self.generator.TASK_SPECS["stack_red_bowl_into_grey_bowl"]
        red = bowls.actors["red_bowl"]
        grey = bowls.actors["grey_bowl"]
        self.assertEqual(red.bounds.x, (0.0, 0.0))
        self.assertEqual(red.bounds.y, (0.0, 0.15))
        self.assertEqual(red.bounds.yaw, (0.0, 0.0))
        self.assertEqual(grey.bounds.x, (-0.10, 0.10))
        self.assertEqual(grey.bounds.y, (0.0, 0.166))
        self.assertEqual(grey.bounds.yaw, (0.0, 0.0))
        self.assertEqual(bowls.constraints[0].kind, "minimum_planar_distance")

    def test_environment_enumerates_json_without_runtime_randomization(self):
        environment_path = (
            ROOT
            / "src"
            / "sim_evals"
            / "environments"
            / "mila"
            / "mila_droid_environment.py"
        )
        source = environment_path.read_text(encoding="utf-8")
        self.assertNotIn("randomize_mila_task_assets", source)
        self.assertIn("func=reset_initial_conditions", source)

    def test_success_geometry_remains_task_specific(self):
        sink = self.tasks["place_spoon_in_sink"]
        self.assertEqual(
            sink.object_region_points,
            ((0.0, -0.18, 0.0), (0.0, 0.18, 0.0)),
        )
        self.assertEqual(sink.target_region_prim_path, "Sink003/Sink003/Sites/reg_basin")

    def test_initial_conditions_use_existing_json_contract(self):
        for task in self.tasks.values():
            path = task.initial_conditions_file
            self.assertEqual(path.name, "initial_conditions.json")
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(raw), {"instruction", "poses"})
            self.assertEqual(raw["instruction"], task.language_instruction)
            self.assertEqual(len(raw["poses"]), 20)
            self.assertEqual(
                raw,
                self.generator.generate_document(task.task_id),
            )
            self.assertEqual(
                len({json.dumps(pose, sort_keys=True) for pose in raw["poses"]}),
                20,
            )
            self.assertEqual(
                set(raw["poses"][0]),
                {task.object_name}
                | ({task.target_name} if task.target_prim_path is not None else set()),
            )
            for pose in raw["poses"][0].values():
                self.assertEqual(len(pose), 7)

            object_pose = raw["poses"][0][task.object_name]
            self.assertEqual(tuple(object_pose[:3]), task.object_init_pos_world)
            self.assertEqual(tuple(object_pose[3:]), task.object_init_rot_world)
            if task.target_prim_path is not None:
                target_pose = raw["poses"][0][task.target_name]
                self.assertEqual(tuple(target_pose[:3]), task.target_init_pos_world)
                self.assertEqual(tuple(target_pose[3:]), task.target_init_rot_world)

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
            bowls.object_init_pos_world[2],
            0.9007219075693947,
        )
        self.assertAlmostEqual(
            bowls.target_init_pos_world[2],
            0.9111174353900667,
        )


if __name__ == "__main__":
    unittest.main()
