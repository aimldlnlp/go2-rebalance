import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
import walking_progress_reward_pilot as study


class Tests(unittest.TestCase):
    def test_only_progress_reward_changes(self):
        for condition in ['healthy','RR_calf']:
            a=study.ProgressEnv(condition,record_trace=True)
            b=study.original.WalkingResidualEnv(condition,record_trace=True)
            try:
                left,_=a.reset(seed=23000)
                right,_=b.reset(seed=23000)
                np.testing.assert_array_equal(left,right)
                if condition!='healthy':
                    for env in [a,b]:
                        env.data.qpos[0]=.501
                for action in [np.linspace(-2,2,12),np.linspace(.7,-.7,12)]:
                    left=a.step(action)
                    right=b.step(action)
                    np.testing.assert_array_equal(left[0],right[0])
                    self.assertEqual(left[2:4],right[2:4])
                    lt,rt=left[4]['reward_terms'],right[4]['reward_terms']
                    self.assertEqual(lt.keys(),rt.keys())
                    for key in lt:
                        self.assertEqual(lt[key],rt[key]*(10 if key=='goal_progress' else 1))
                    self.assertEqual(left[1],sum(lt.values()))
                    for key in a.trace():
                        np.testing.assert_array_equal(a.trace()[key],b.trace()[key])
                self.assertIs(study.ProgressEnv._observation,study.original.WalkingResidualEnv._observation)
            finally:
                a.close()
                b.close()

    def test_environment_and_contract(self):
        env=study.ProgressEnv('healthy')
        try:
            check_env(env,skip_render_check=True)
        finally:
            env.close()
        delta=[k for k in study.REWARD_WEIGHTS if study.REWARD_WEIGHTS[k]!=study.original.REWARD_WEIGHTS[k]]
        self.assertEqual(delta,['goal_progress'])
        self.assertEqual(study.REWARD_WEIGHTS['goal_progress'],100.)


if __name__=='__main__':
    unittest.main()
