# Third-party references

The project builds on these upstream projects. Their source assets, pretrained checkpoints and license notices are not bundled in this curated repository.

- [MuJoCo](https://github.com/google-deepmind/mujoco): simulator.
- [MuJoCo Menagerie, Unitree Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2): robot model and visual assets. Recorded revision: `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`. The local Go2 model includes its BSD-3-Clause notice.
- [Go2 RL Lab](https://github.com/wty-yy/go2_rl_lab): nominal walking policy and deployment contract. Recorded revision: `6c95d9feef4814c25eb6536e61bb38e1d1d718d4`. The local upstream repository notice contains MIT/Apache terms; no separate checkpoint-specific license declaration was recorded.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): PPO and policy serialization.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): environment interface.

Earlier exploratory code also references [go2-sim2real-deploy](https://github.com/saifahmadgit/go2-sim2real-deploy), revision `92b847b2f4e5b5974846d99f3ce2a494c242c121`, with a BSD-3-Clause repository notice. That checkpoint is not the retained nominal walker.
