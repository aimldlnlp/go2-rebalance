"""One RR-only training-allocation pilot; original reward and matched pose RNG."""
from project_paths import project_file, source_files
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import numpy as np
import walking_progress_reward_pilot as runner
import walking_residual_privileged_env as original
import walking_residual_privileged_pilot as parent
import sudden as s

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/walking_residual_privileged_pilot'
OUT=ROOT/'results/walking_rr_specialist_pilot'


class SpecialistEnv(original.WalkingResidualEnv):
    def reset(self,*,seed=None,options=None):
        requested=(options or {}).get('condition',self.condition_mode)
        obs,info=super().reset(seed=seed,options=options)
        if requested=='mixed':
            # Preserve the parent's condition draw and pose jitter RNG sequence.
            self.condition='RR_calf'
            self.weak=self.model.actuator('RR_calf').id
            obs,info=self._observation(),self._info()
        return obs,info


def baseline_parity(rows):
    assert len(rows)==25
    for row in rows:
        stem=f"{row['condition']}_{row['seed']}"
        old=s.read(SOURCE/'baseline'/(stem+'.json'))
        with np.load(OUT/'baseline'/(stem+'.npz'),allow_pickle=False) as a,np.load(SOURCE/'baseline'/(stem+'.npz'),allow_pickle=False) as b:
            assert a.files==b.files
            for key in b.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        for key in ['initial_state_sha256','success','fallen','fault_onset_tick','final_progress_m','final_stable_hold_s','episode_return']:
            assert row[key]==old[key],key
    s.write(OUT/'baseline_parity.json',dict(passed=True,episodes=25,all_original_arrays_exact=True,rewards_exact=True))


def prepare():
    assert not OUT.exists(),'Never overwrite an existing pilot'
    assert 'OK' in (ROOT/'work/rr_specialist_tests.log').read_text(encoding='utf-8')
    assert s.read(SOURCE/'final_verification.json')['audit_passed']
    design=s.read(SOURCE/'design.json')
    for name,sha in design['source_sha256'].items():assert s.digest(project_file(ROOT, name))==sha,name
    old=s.read(SOURCE/'protocol.json')
    files=[p for p in (ROOT/'results').rglob('*') if p.is_file()]+list(source_files(ROOT))
    files+=[p for p in (ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*') if p.is_file()]
    files+=[ROOT/'vendor/walking_baselines/go2_rl_lab/policy.pt',ROOT/'work/rr_specialist_tests.log']
    hashes={}
    for path in files:
        hashes[str(path.relative_to(ROOT))]=s.digest(path)
        if len(hashes)%500==0:print(f'Protected hashes: {len(hashes)}',flush=True)
    p={**old,'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'One RR-only training-allocation pilot; frozen before training',
        'only_change':'Training condition distribution: uniform healthy/FL/FR/RL/RR to RR-only. Validation distribution unchanged.',
        'only_controller_change':'No observation, action, architecture, controller, reward or dynamics changes.',
        'reward_weights':dict(original.REWARD_WEIGHTS),
        'training_condition':'RR_calf','fault_strength_remaining':.1,
        'control':'Archived privileged mixed-condition seed0 pilot; original goal-progress10 reward',
        'initialization':'Fresh PPO seed0; weights/sigma/update counters must exactly match control initial_policy.json',
        'pose_randomization':'Keep original mixed reset and its RNG draws, then set the selected training fault to RR. Same seed yields identical pre-fault physical initial state and RNG state.',
        'max_training_control_steps':131072,'wall_time_cap_s':1800,'checkpoints':[32768,65536,131072],
        'validation_seeds':list(range(23000,23005)),'confirmation_seeds':[],
        'confirmation_policy':'No confirmation or final test, even if development target gate passes',
        'primary_endpoints':['RR mission success','RR falls','healthy mission success','healthy falls'],
        'specialist_development_gate':'Healthy5/5 and RR>=3/5, no falls/interruption within those10 episodes',
        'general_development_gate':'Original full gate retained in acceptance files: healthy5/5, each of4 faults>=3/5, zero falls; never promotes without confirmation',
        'parity':'25 zero-residual baseline replays, every array including rewards exact before PPO updates',
        'scope':'One training run and seed, fixed131072 total steps, RR allocation includes pre-fault controls; no curriculum, fine tuning, student, extra training seeds or automatic budget extension',
        'interpretation':'Tests allocation of a fixed total budget. More RR exposure and removal of other conditions are inherent in this intervention; does not isolate specialization from exposure or establish generalization.'}
    for key in ['hyperparameters','ppo_fixed_defaults','reset_seed_pool','training_seed','observation_shape','observation_fields']:
        assert p[key]==old[key],key
    assert p['reward_weights']==original.REWARD_WEIGHTS and p['reward_weights']['goal_progress']==10.
    OUT.mkdir();s.write(OUT/'protocol.json',p)
    s.write(OUT/'design.json',dict(protocol_sha256=s.digest(OUT/'protocol.json'),protected_inputs=hashes))
    shutil.copyfile(ROOT/'work/rr_specialist_tests.log',OUT/'tests-before.log')
    print(f'Frozen RR specialist131072/1800s; {len(hashes)} protected files',flush=True)


def verify(full=False):
    p=s.read(OUT/'protocol.json');design=s.read(OUT/'design.json')
    assert s.digest(OUT/'protocol.json')==design['protocol_sha256']
    for name,sha in design['protected_inputs'].items():
        if full or name.endswith('.py'):assert s.digest(project_file(ROOT, name))==sha,name
    assert p['reward_weights']==original.REWARD_WEIGHTS
    return p


def run():
    runner.OUT=OUT;runner.SOURCE=SOURCE
    runner.ProgressEnv=SpecialistEnv
    runner.REWARD_WEIGHTS=original.REWARD_WEIGHTS
    runner.verify=verify;runner.baseline_parity=baseline_parity
    runner.run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args();{'prepare':prepare,'run':run}[args.command]()
