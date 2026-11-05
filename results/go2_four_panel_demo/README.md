# All-motor four-panel demo

`go2_all_motors_four_panel_60s.mp4`: 60 seconds, 1920×1080, 25 fps, English, silent. It shows all thirteen tested conditions using one shared development pose, 23003. No opening title card, summary, controller diagram or playback-speed caption is inserted.

## Sequence and layout

- 0–5.00 seconds: the complete healthy mission, fullscreen.
- 5.00–23.32 seconds: four hip-motor faults simultaneously.
- 23.32–41.68 seconds: four thigh-motor faults simultaneously.
- 41.68–60.00 seconds: four knee-motor faults simultaneously.

Panel positions remain fixed: front left at top left, front right at top right, rear left at bottom left and rear right at bottom right. All four panels within each section use the same saved simulation time and joint type. All twelve faults are separate episodes, not simultaneous failures on one robot.

Every source episode is shown continuously from simulation time 0.00 through 20.00 seconds. The healthy episode occupies five video seconds; each fault episode occupies about 18.33 video seconds. The graph x-axis displays original simulation time, not video time. There are only three layout/condition-group transitions, and the video ends on the final mission frames.

## Data shown

The arena fills each panel. Overlays show the leg, joint and remaining torque fraction, a live graph of actual 2D goal distance against simulation time, and the recorded stable-hold counter near the target. Path, lateral limits and goal ring are drawn in world coordinates. The actual weakened joint changes to amber at its recorded fault tick. Internal actuator name `calf` is labeled “Knee”.

The goal-error y-axis is shared across all four panels within a joint section. It changes between sections to cover their full recorded ranges: inspect `graph_high_by_segment` in `video_plan.json`. Both failed and successful trajectories remain visible without clipping. The dashed reference represents 0.15 m goal error. Meeting that line or the hold indicator alone does not certify every mission criterion.

These are thirteen example episodes, not all 65 development episodes. The retained candidate index has healthy 5/5, eight motor conditions meeting at least 3/5, and four unresolved front hip/knee conditions. The displayed shared pose is not selected separately to make each motor look successful. Results do not establish generalization or hardware performance. Fault identity is supplied by the simulator.

## Reuse and checks

`research/make_four_panel_demo.py` reuses the saved-state renderer, fonts, data derivation, goal projection and encoder from the earlier videos. It never integrates physics, predicts actions or trains a policy. Initial states and pre-fault qpos/qvel prefixes are exactly paired across the thirteen conditions. All thirteen sources received independent physics, fault, torque and mission verification before rendering. The subsequent directory cleanup changed source paths; historical render hashes describe their original versions.

`video_plan.json` freezes selections, timing, layout and source hashes. `frame_indices.json` maps each encoded frame to every displayed source condition and tick. `render_audit.json` verifies 5,625 exact pose draws and actual-joint highlighting. `verification.json` checks all thirteen complete source intervals, the four-panel mappings, full-stream MP4 decode, size, frame count, silence and protected hashes. Decoded stills and `contact_sheet.png` support sampled visual review.

At the owner's request, this is now the only video export retained in the project. Twelve earlier exports were deleted during repository cleanup. Their historical plans may still reference those files; these are records of the original production state. This MP4 remains byte-for-byte unchanged. Use a new output directory for future revisions; prepare/render refuse to overwrite completed artifacts.
