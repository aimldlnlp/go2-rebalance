import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import walking_baseline as old
import walking_calibrated as fixed

class CalibrationTests(unittest.TestCase):
    def test_matches_source_passives_preserving_other_parameters(self):
        before=mujoco.MjModel.from_xml_path(str(old.MODEL));after=fixed.calibrated_model()
        for name in ['jnt_range','actuator_ctrlrange','body_mass','body_inertia','geom_friction','geom_size','dof_armature']:
            np.testing.assert_array_equal(getattr(before,name),getattr(after,name))
        self.assertEqual(before.opt.timestep,after.opt.timestep)
        np.testing.assert_array_equal(after.dof_damping[6:],np.zeros(12))
        np.testing.assert_array_equal(after.dof_frictionloss[6:],np.full(12,.01))
        np.testing.assert_array_equal(before.dof_damping[6:],np.full(12,2.))
        np.testing.assert_array_equal(before.dof_frictionloss[6:],np.full(12,.2))
    def test_preserves_mission_helpers_and_fresh_split(self):
        self.assertIs(fixed.summarize,old.summarize)
        self.assertIs(fixed.mission_command,old.mission_command)
        self.assertIs(fixed.torque_limit,old.torque_limit)
        self.assertFalse(set(fixed.HEALTHY_SEEDS)&set(old.HEALTHY_SEEDS))
    def test_calibrated_step_uses_same_limits(self):
        with patch.object(fixed,'TICKS',2):
            data,row=fixed.episode(old.Actor('go2_rl_lab'),987649)
        self.assertFalse(row['success'])
        np.testing.assert_allclose(data['actuator_force'],data['applied_torque'],rtol=0,atol=1e-12)
        self.assertTrue(np.all(np.abs(data['limited_torque'])<=np.array([23.7,23.7,45.43]*4)))

if __name__=='__main__':unittest.main()
