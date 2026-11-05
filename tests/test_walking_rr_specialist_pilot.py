import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
import walking_rr_specialist_pilot as study


class Tests(unittest.TestCase):
    def test_training_pose_and_rng_are_matched(self):
        a=study.SpecialistEnv('mixed');b=study.original.WalkingResidualEnv('mixed')
        try:
            for seed in [24000,24001,24004,24050]:
                left,_=a.reset(seed=seed,options={'condition':'mixed'})
                right,_=b.reset(seed=seed,options={'condition':'mixed'})
                self.assertEqual(a.condition,'RR_calf');self.assertEqual(a.weak,11)
                np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
                np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
                np.testing.assert_array_equal(left,right)
                self.assertEqual(a.np_random.bit_generator.state,b.np_random.bit_generator.state)
                np.testing.assert_array_equal(left[60:],np.ones(12))
        finally:a.close();b.close()

    def test_validation_and_dynamics_reward_unchanged(self):
        a=study.SpecialistEnv('healthy',record_trace=True)
        b=study.original.WalkingResidualEnv('healthy',record_trace=True)
        try:
            for condition in study.original.CONDITIONS:
                left,_=a.reset(seed=23000,options={'condition':condition})
                right,_=b.reset(seed=23000,options={'condition':condition})
                self.assertEqual(a.condition,condition)
                np.testing.assert_array_equal(left,right)
                for env in [a,b]:env.data.qpos[0]=.501
                left=a.step(np.linspace(-2,2,12));right=b.step(np.linspace(-2,2,12))
                np.testing.assert_array_equal(left[0],right[0])
                self.assertEqual(left[1:],right[1:])
                for key in a.trace():np.testing.assert_array_equal(a.trace()[key],b.trace()[key])
            self.assertIs(study.SpecialistEnv.step,study.original.WalkingResidualEnv.step)
            self.assertIs(study.SpecialistEnv._observation,study.original.WalkingResidualEnv._observation)
        finally:a.close();b.close()

    def test_environment(self):
        env=study.SpecialistEnv('mixed')
        try:check_env(env,skip_render_check=True)
        finally:env.close()


if __name__=='__main__':unittest.main()
