from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import controllers
from controllers.controller_api import PersistentOvershootCap, validate_controller_config
from controllers.controller_example import DEFAULT_CHECKPOINT
from controllers.new_pf_controller import RobustPFController
from controllers.new_rl_controller import PPOVolumeController
from controllers.new_rl_numpy_controller import NumpyPPOVolumeController
from controllers.particle_inference import FixedKParticleFilter, PKA_CLIP_HIGH, PKA_CLIP_LOW
from controllers.chemistry_model import SolutionState
from training.task_distribution import generate_tasks
from training.models import load_actor_checkpoint
from scripts.train_pipeline import PROFILES


class ReleaseContractTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        self.assertTrue(hasattr(controllers, "RobustPFController"))

    def test_invalid_controller_configuration_is_rejected(self) -> None:
        invalid = (
            {"success_tolerance_ph": -0.1},
            {"max_steps": 0},
            {"max_total_dose_ml": -1.0},
            {"titrant_concentration_m": 0.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RobustPFController(**kwargs)

    def test_particle_filter_rejects_invalid_fixed_concentration(self) -> None:
        from controllers.particle_inference import FixedKParticleFilter

        with self.assertRaises(ValueError):
            FixedKParticleFilter(20, 1, False, np.random.default_rng(0), fixed_concentration_m=0.0)

    def test_action_and_delivery_respect_total_dose_limit(self) -> None:
        controller = RobustPFController(particles=1000, seed=7, max_total_dose_ml=0.05)
        controller.reset(4.0, 8.0, 10.0)
        action = controller.recommend()
        self.assertLessEqual(action.volume_ml, 0.05)
        with self.assertRaises(ValueError):
            controller.observe(4.1, actual_volume_ml=0.06, reagent=action.reagent)

    def test_persistent_overshoot_cap_is_shared_and_monotonic(self) -> None:
        cap = PersistentOvershootCap()
        uncapped, applied = cap.apply(5.0)
        self.assertEqual((uncapped, applied), (5.0, False))

        self.assertTrue(cap.update(4.0, 8.0, 6.0, 5.0))
        self.assertEqual(cap.cap_ml, 2.5)
        capped, applied = cap.apply(5.0)
        self.assertEqual((capped, applied), (2.5, True))

        # A later trigger with a larger delivered dose cannot loosen the cap.
        self.assertTrue(cap.update(8.0, 9.0, 6.0, 8.0))
        self.assertEqual(cap.cap_ml, 2.5)

        # An error increase can trigger the cap without crossing the target.
        self.assertTrue(cap.update(5.0, 4.0, 6.0, 1.0))
        self.assertEqual(cap.cap_ml, 0.5)
        self.assertEqual(cap.apply(0.9), (0.5, True))

        cap.reset()
        self.assertIsNone(cap.cap_ml)
        self.assertEqual(cap.events, 0)
        self.assertEqual(cap.applied_steps, 0)

    def test_both_ppo_deployment_backends_apply_persistent_cap(self) -> None:
        controllers_to_check = (
            PPOVolumeController(ROOT / "controllers" / "models" / "ppo_seed_303.pth", device="cpu"),
            NumpyPPOVolumeController(ROOT / "controllers" / "models" / "ppo_seed_303_numpy.npz"),
        )
        for controller in controllers_to_check:
            with self.subTest(controller=type(controller).__name__):
                controller.reset(4.0, 8.0)
                controller.overshoot_cap.update(4.0, 9.0, 8.0, 0.02)
                action = controller.recommend()
                self.assertEqual(action.volume_ml, 0.01)
                self.assertTrue(action.diagnostics["overshoot_cap_applied"])
                controller.reset(4.0, 8.0)
                self.assertIsNone(controller.status()["overshoot_cap_ml"])

    def test_default_checkpoint_is_independent_of_working_directory(self) -> None:
        self.assertTrue(DEFAULT_CHECKPOINT.is_absolute())
        self.assertTrue(DEFAULT_CHECKPOINT.is_file())

    def test_pka_rejuvenation_is_clipped_to_documented_envelope(self) -> None:
        particle_filter = FixedKParticleFilter(
            particle_count=20,
            pair_count=1,
            infer_concentration=False,
            rng=np.random.default_rng(1),
        )
        particle_filter.pka_particles[:] = np.linspace(-20.0, 20.0, 20)[:, None]
        particle_filter.weights.fill(0.0)
        particle_filter.weights[0] = 1.0
        particle_filter.update(
            11.0,
            SolutionState(11.0, 0.0, 0.0),
            SolutionState(11.0, 0.0001, 0.0),
            2.0,
            2.1,
        )
        self.assertGreaterEqual(float(np.min(particle_filter.pka_particles)), PKA_CLIP_LOW)
        self.assertLessEqual(float(np.max(particle_filter.pka_particles)), PKA_CLIP_HIGH)

    def test_task_generation_is_deterministic_for_a_seed(self) -> None:
        first = generate_tasks(123, 5, "test")
        second = generate_tasks(123, 5, "test")
        self.assertEqual(first, second)

    def test_standard_profile_uses_published_validation_sizes(self) -> None:
        self.assertEqual(PROFILES["standard"]["validation_tasks"], 500)
        self.assertEqual(PROFILES["standard"]["imitation_validation_tasks"], 500)
        self.assertEqual(PROFILES["standard"]["ppo_validation_tasks"], 500)

    def test_smoke_profile_is_explicitly_separate(self) -> None:
        self.assertIn("smoke", PROFILES)
        self.assertNotIn("quick", PROFILES)

    def test_controller_status_identifies_protocol_profile(self) -> None:
        controller = NumpyPPOVolumeController(
            ROOT / "controllers" / "models" / "ppo_seed_303_numpy.npz"
        )
        status = controller.reset(4.0, 8.0)
        self.assertEqual(status["protocol"]["protocol_family"], "pH-control")
        self.assertEqual(status["protocol"]["protocol_version"], "2026.08")
        self.assertEqual(status["protocol"]["protocol_profile"], "deployment_api_strict")
        self.assertEqual(status["protocol"]["controller_stop_operator"], "<")

    def test_training_checkpoint_hash_is_checked_before_load(self) -> None:
        checkpoint = ROOT / "controllers" / "models" / "ppo_seed_303.pth"
        with self.assertRaisesRegex(RuntimeError, "Checkpoint hash mismatch"):
            load_actor_checkpoint(checkpoint, "cpu", expected_file_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
