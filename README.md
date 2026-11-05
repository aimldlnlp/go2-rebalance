# Go2 Rebalance

Go2 Rebalance tests whether a Unitree Go2 can finish a short walk after one motor loses most of its torque. The robot has to travel 2 m within 20 seconds and stop steadily for at least 2 seconds.

At 0.50 m, one motor is reduced to 10% of its torque after actuator limits. The other eleven motors stay at full strength. Every hip, thigh and knee motor is tested separately in a flat MuJoCo arena, with simulation and training running on CPU.

https://github.com/user-attachments/assets/5098d507-3edc-4f59-9bc2-7a04b6014130

The one-minute demo starts with healthy control, then shows four legs side by side while cycling through hip, thigh and knee faults. Each panel is a separate run with one weak motor. All thirteen conditions use the same starting pose. The small graphs show distance to the goal over simulation time.

## Results so far

Eight of the twelve motor-fault conditions reach the target of at least three successful walks out of five. The front hips and front knees still fall short.

| Weakened leg | Hip | Thigh | Knee |
|---|---:|---:|---:|
| Front left | 0/5 | 3/5 | 0/5 |
| Front right | 0/5 | 3/5 | 0/5 |
| Rear left | 3/5 | 4/5 | 3/5 |
| Rear right | 3/5 | 4/5 | 4/5 |

Healthy control succeeds in all five runs. Across healthy control and the twelve fault conditions, 32 of 65 missions succeed and none end in a fall. These results combine development runs collected over several experiments. The five starting poses were reused while tuning, so they do not measure performance on unseen conditions.

The video shows one pose per condition. [Per-episode measurements](docs/results.json) cover all 65 runs.

## What counts as a successful walk?

The robot must complete the full 20-second episode without falling and meet every limit below:

| Measurement | Required result |
|---|---|
| Final 2D position error | At most 0.15 m |
| Final heading error | At most 15° |
| Largest lateral deviation during the walk | At most 0.20 m |
| Continuous stop at the end | At least 2 s |
| Planar speed during that stop | At most 0.05 m/s |

The stop only counts while the robot is inside the position and heading limits. Fault runs also require the motor to weaken before the robot reaches the goal. Healthy control must pass all five starting poses; each weak motor must pass at least three, with no falls across the tests.

## How it works

A pretrained walking policy supplies the nominal joint targets. A PPO policy adds small residual corrections, and PD control turns the targets into motor torques. Navigation handles the 2 m target and the final stop. Additional adjustments depend on which motor is weak.

The simulator supplies motor-strength information to the controller. Detecting a fault from real sensor data is not part of these tests. The current controller also has not reached the target for every motor.

The remaining failures have different shapes. Front hip faults can send the robot sideways and then backward. Front knee faults lower the body and reduce forward progress before the goal. Those observations guide the next changes; simply getting closer to the goal does not count as completing the mission.

## Code

Research code lives in `research/`, tests in `tests/`, and supporting notes in `docs/`.

| Start here | What it does |
|---|---|
| `research/walking_baseline.py` | Adapts the pretrained walker and applies the torque-speed limits |
| `research/walking_residual_env.py` | Runs the residual controller and records the physical state |
| `research/walking_navigation_hold.py` | Steers toward the target and latches the stop command |
| `research/review_walking.py` | Recomputes mission and torque checks from saved traces |
| `research/make_four_panel_demo.py` | Renders the demo from recorded poses |

Earlier standing-control experiments remain in `research/`. Their results are separate from the walking tests reported here.

## Running the code

The recorded environment uses Python 3.12, CPU PyTorch, MuJoCo and Stable-Baselines3. [requirements-lock.txt](requirements-lock.txt) records the installed versions. Follow the [setup notes](docs/REPRODUCIBILITY.md) to obtain the robot model, policy assets and other inputs.

After installing dependencies:

```powershell
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
.venv\Scripts\python.exe -m unittest tests.test_project_paths tests.test_go2 tests.test_walking_baseline -v
```

These checks require the model and nominal-policy assets. They do not train a policy. Raw traces, checkpoints and third-party assets are kept outside Git, so a fresh clone does not contain everything needed to repeat training or rerender the experiments. Some older scripts also expect their original local files; the setup notes describe those dependencies.

The complete walking controller still needs to pass development testing before evaluation on new starting poses. Physical robot tests, uneven terrain and multiple simultaneous motor faults remain untested.

[Third-party references](docs/THIRD_PARTY.md) list the simulator, Go2 model and pretrained-policy sources.
