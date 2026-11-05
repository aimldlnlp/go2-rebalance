"""Render saved walking states directly. No integration, inference or evaluation."""
from project_paths import project_file, source_files
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw

import sudden as s
import diagnose_walking_fault as d
import make_multimotor_video as previous
import walking_rr_severity as study

ROOT=s.ROOT
OUT=ROOT/'results/walking_rr_severity_video'
DRAFT=ROOT/'work/walking_rr_severity_video_previews_final'
FPS=25
SIZE=(1920,1080)
ORDER=['healthy','rr_75','rr_50','rr_25','rr_10']
LABELS={'healthy':'Healthy','rr_75':'RR 75%','rr_50':'RR 50%','rr_25':'RR 25%','rr_10':'RR 10%'}
CHAPTERS=[('mission',10),('paired',20),('levels',10),('late_arrival',8),('summary',12)]


def media_helper():
    spec=importlib.util.spec_from_file_location('rr_saved_pose_media',previous.HELPER)
    media=importlib.util.module_from_spec(spec);spec.loader.exec_module(media)
    assert media.font(32).getname()[0]=='Latin Modern Roman'
    return media


def sample_tick(frame,rate=1,start_s=0):
    return min(10000,round(start_s/.002)+frame*round(rate/(FPS*.002)))


def load_cases():
    assert s.read(study.OUT/'completion_audit.json')['study_complete']
    records=s.read(study.OUT/'historical_controls.json')+s.read(study.OUT/'episodes.json')
    summaries=s.read(study.OUT/'summary.json')
    cases={}
    for name,strength in zip(ORDER,[1.,.75,.5,.25,.1]):
        cohort=[r for r in records if r['strength_remaining']==strength]
        summary=next(g for g in summaries if g['strength_remaining']==strength)
        assert len(cohort)==5 and sorted(r['seed'] for r in cohort)==list(range(23000,23005))
        assert summary['successes']==sum(r['success'] for r in cohort)
        assert summary['falls']==sum(r['fallen'] for r in cohort)
        for field,source in [('median_goal_error_m','final_goal_error_m'),('median_stable_hold_s','final_stable_hold_s')]:
            np.testing.assert_allclose(summary[field],np.median([r[source] for r in cohort]),rtol=0,atol=1e-14)
        row=next(r for r in records if r['strength_remaining']==strength and r['seed']==23000)
        path=ROOT/row['trace'];assert s.digest(path)==row['trace_sha256']
        with np.load(path,allow_pickle=False) as archive:
            q,v=archive['qpos'].copy(),archive['qvel'].copy()
            commands=archive['command'].copy()
            assert not np.any(archive['control_residual_actions'])
        assert q.shape==(10001,19) and v.shape==(10001,18)
        assert np.isfinite(q).all() and np.isfinite(v).all()
        assert s.digest_bytes(q[0].tobytes()+v[0].tobytes())==row['initial_state_sha256']
        f=np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
        delta=q[:,:2]-row['initial_xy'];progress=delta@f
        error=np.linalg.norm(delta-2*f,axis=1)
        qw,qx,qy,qz=q[:,3:7].T
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        heading=np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
        stable=(error<=.15)&(heading<=np.deg2rad(15))&(np.linalg.norm(v[:,:2],axis=1)<=.05)
        hold=np.zeros(len(q));begin=None
        for i,ok in enumerate(stable):
            if not ok:begin=None
            elif begin is None:begin=i
            hold[i]=0 if begin is None else (i-begin)*.002
        np.testing.assert_allclose(hold[-1],row['final_stable_hold_s'],atol=1e-9,rtol=0)
        goal=np.flatnonzero(error<=.15)
        cases[name]=dict(row=row,qpos=q,qvel=v,commands=commands,progress=progress,error=error,hold=hold,
            first_goal_entry_s=float(goal[0]*.002) if len(goal) else None,
            summary=next(g for g in summaries if g['strength_remaining']==strength))
    for name in ORDER:
        np.testing.assert_array_equal(cases[name]['qpos'][0],cases['healthy']['qpos'][0])
        np.testing.assert_array_equal(cases[name]['qvel'][0],cases['healthy']['qvel'][0])
    return cases


def pose(media,studio,case,tick,size):
    active=case['row']['condition']!='healthy' and tick>=case['row']['fault_onset_tick']
    previous.highlight(studio,studio.original,'RR_calf' if active else 'healthy')
    studio.camera.lookat[:]=[case['qpos'][tick,0],case['qpos'][tick,1],.20]
    studio.data.qvel[:]=case['qvel'][tick]
    image=studio.frame(case['qpos'][tick])
    np.testing.assert_array_equal(studio.data.qpos,case['qpos'][tick])
    np.testing.assert_array_equal(studio.data.qvel,case['qvel'][tick])
    studio.drawn.append((case['row']['strength_remaining'],tick))
    return image.resize(size,Image.Resampling.LANCZOS)


def studio_for(media):
    studio=media.Studio()
    assert (studio.model.nq,studio.model.nv,studio.model.nu)==(19,18,12)
    assert [studio.model.actuator(i).name for i in range(12)]==study.baseline.JOINTS
    studio.original=mujoco.MjModel.from_xml_path(str(media.MODEL))
    studio.camera.distance=.94
    studio.drawn=[]
    return studio


def progress_rail(media,image,case,tick,x,y,width):
    draw=ImageDraw.Draw(image)
    draw.line((x,y,x+width,y),fill='#c5cabf',width=2)
    for value in [0,1,2]:
        at=x+width*value/2
        draw.line((at,y-5,at,y+5),fill=media.MUTED,width=1)
        media.put(image,str(value),at,y+10,23,center=True)
    at=x+width*np.clip(case['progress'][tick]/2,0,1)
    draw.ellipse((at-6,y-6,at+6,y+6),fill=media.SAGE if case['row']['condition']=='healthy' else media.AMBER)


def opening(media,studio,cases):
    image=media.page('Walking with a weaker rear-right knee','A frozen nominal policy, with no residual corrections.')
    media.put(image,'The mission',80,246,48,width=960)
    lines=['Move 2 m forward within 20 s.','Command speed: at most 0.20 m/s.',
           'Final error: at most 0.15 m; heading: at most 15 degrees.',
           'Maximum cross-track error: 0.20 m.',
           'Hold position and heading for the final 2 s,',
           'with planar speed at most 0.05 m/s.',
           'Fall limits: height below 0.16 m or tilt above 45 degrees.',
           'Any fall or nonfinite state fails the mission.']
    for i,line in enumerate(lines):media.put(image,line,80,330+i*55,29,width=1030)
    image.paste(pose(media,studio,cases['healthy'],0,(700,400)),(1120,265))
    media.put(image,'RR_calf is the rear-right knee actuator.',1120,700,28,width=720)
    media.put(image,'The fault latches at 0.50 m of progress.\nIts torque is scaled after torque limiting.',1120,755,28,width=720)
    media.put(image,'Pose 23000 is shown throughout. The closing table covers all five poses.',80,900,32,width=1760)
    media.put(image,'Healthy and RR 10% are archived controls. Motion comes directly from saved states.',80,1008,27,fill=media.MUTED,width=1760)
    return image


def paired(media,studio,cases,tick):
    image=media.page('Healthy and RR 75%','Same starting pose 23000. Tracking cameras. Playback at normal speed.')
    ImageDraw.Draw(image).line((960,230,960,982),fill='#c5cabf')
    for name,x in [('healthy',80),('rr_75',1000)]:
        c=cases[name];color=media.SAGE if name=='healthy' else media.AMBER
        media.put(image,LABELS[name],x,236,42,fill=color,width=840)
        media.put(image,'Archived control' if c['row']['historical_control'] else 'Saved severity run',x,292,27,fill=media.MUTED,width=840)
        image.paste(pose(media,studio,c,tick,(840,480)),(x,333))
        progress_rail(media,image,c,tick,x+50,835,700)
        media.put(image,f"Progress: {c['progress'][tick]:.2f} m",x,896,31,width=840)
        media.put(image,f"Position error: {c['error'][tick]:.2f} m    Hold: {c['hold'][tick]:.2f} s",x,940,29,width=840)
    c=cases['rr_75'];time_s=tick*.002
    caption='Both runs begin at full motor strength.' if time_s<c['row']['fault_onset_tick']*.002 else 'RR torque is now 75% of its limited value.'
    if time_s>=c['first_goal_entry_s']:caption='RR 75% enters the goal area after 18 s, leaving too little hold time.'
    media.put(image,caption,80,1015,28,width=1550)
    media.put(image,f't = {time_s:05.2f} s',1650,1015,28,width=250)
    return image


def levels(media,studio,cases,tick):
    image=Image.new('RGB',SIZE,'white')
    media.put(image,'Four levels of rear-right torque',70,25,62,width=1760)
    media.put(image,'Same saved pose 23000. All residual actions are zero. Playback at 2x speed.',70,112,30,fill=media.MUTED,width=1780)
    draw=ImageDraw.Draw(image);draw.line((960,185,960,1007),fill='#c5cabf');draw.line((70,605,1850,605),fill='#c5cabf')
    for i,name in enumerate(ORDER[1:]):
        x,y=70+(i%2)*960,190+(i//2)*425;c=cases[name]
        media.put(image,LABELS[name],x,y,36,fill=media.AMBER,width=850)
        active=tick>=c['row']['fault_onset_tick']
        source='Archived control' if c['row']['historical_control'] else 'Saved severity run'
        status=f"{c['row']['strength_remaining']:.0%} after fault" if active else '100% before fault'
        media.put(image,source+' / '+status,x,y+45,24,fill=media.MUTED,width=850)
        image.paste(pose(media,studio,c,tick,(560,320)),(x,y+78))
        media.put(image,f"{c['progress'][tick]:.2f} m",x+710,y+132,52,center=True,fill=media.AMBER)
        media.put(image,'Forward progress',x+710,y+204,25,center=True)
        media.put(image,f"{c['error'][tick]:.2f} m",x+710,y+266,38,center=True)
        media.put(image,'Position error',x+710,y+317,25,center=True)
    media.put(image,'Torque is scaled after limiting, on RR_calf only.',70,1035,25,fill=media.MUTED,width=1370)
    media.put(image,f't = {tick*.002:05.2f} s',1660,1035,25,width=230)
    return image


def late_arrival(media,studio,cases,tick):
    c=cases['rr_75']
    image=media.page('RR 75% arrives too late for the hold','Pose 23000. The final 2 s must meet position, heading and speed limits.')
    image.paste(pose(media,studio,c,tick,(1008,576)),(80,246))
    media.put(image,'Goal area entered',1190,265,36,width=650)
    media.put(image,f"{c['first_goal_entry_s']:.2f} s",1190,320,74,fill=media.AMBER,width=650)
    media.put(image,'Stable hold at the deadline',1190,455,34,width=650)
    media.put(image,f"{c['row']['final_stable_hold_s']:.2f} s",1190,510,74,fill=media.AMBER,width=650)
    media.put(image,'2.00 s required',1190,610,32,width=650)
    media.put(image,'Position, heading and cross-track passed.\nThe final hold failed.',1190,705,29,width=650)
    x0,x1,y=110,1800,900
    draw=ImageDraw.Draw(image)
    draw.rectangle((x0+(x1-x0)*.5,y-14,x1,y+14),fill='#e9ece6')
    draw.line((x0,y,x1,y),fill=media.MUTED,width=2)
    for t in [16,18,20]:
        x=x0+(x1-x0)*(t-16)/4;draw.line((x,y-8,x,y+8),fill=media.MUTED,width=2);media.put(image,f'{t} s',x,y+30,29,center=True)
    x=x0+(x1-x0)*(tick*.002-16)/4
    draw.ellipse((x-9,y-9,x+9,y+9),fill=media.AMBER)
    footer='Final frame held for reading. Shading marks the required final 2 s.' if tick==10000 else 'The shaded interval is the required final 2 s.'
    media.put(image,footer,80,1008,28,fill=media.MUTED,width=1400)
    media.put(image,f't = {tick*.002:05.2f} s',1650,1008,28,width=250)
    return image


def closing(media,cases):
    image=media.page('Results across all five starting poses','Poses 23000 to 23004. Five saved runs per condition, each lasting 20 s.')
    xs=[80,780,1100,1430,1730]
    for x,text in zip(xs,['Condition','Success','Falls','Median error','Median hold']):media.put(image,text,x,250,33,center=x!=80,width=650 if x==80 else 330)
    draw=ImageDraw.Draw(image);draw.line((80,311,1840,311),fill='#aeb4a8')
    for i,name in enumerate(ORDER):
        y=345+i*92;c=cases[name];g=c['summary']
        label=LABELS[name]+(' (archive)' if g['historical_control'] else '')
        color=media.SAGE if name=='healthy' else media.INK
        media.put(image,label,80,y,36,fill=color,width=650)
        for x,text in zip(xs[1:],[f"{g['successes']}/5",f"{g['falls']}/5",f"{g['median_goal_error_m']:.3f} m",f"{g['median_stable_hold_s']:.2f} s"]):media.put(image,text,x,y,38,center=True,fill=color)
    draw.line((80,825,1840,825),fill='#aeb4a8')
    media.put(image,'RR 75% reached the goal area in all five poses, but too late for the 2 s hold.',80,862,32,width=1760)
    media.put(image,'RR 50%, 25% and 10% missed the position tolerance. No run fell.',80,916,31,width=1760)
    media.put(image,'Flat-floor simulation. Archived controls are labelled. No new runs or training were used for this film.',80,1008,26,fill=media.MUTED,width=1760)
    return image


def preview():
    assert not DRAFT.exists(),'Refusing to overwrite design previews'
    DRAFT.mkdir();media=media_helper();cases=load_cases();studio=studio_for(media)
    try:
        pages={'intro':opening(media,studio,cases),'paired':paired(media,studio,cases,9600),
            'levels':levels(media,studio,cases,8500),'late':late_arrival(media,studio,cases,10000),'summary':closing(media,cases)}
        for name,im in pages.items():im.save(DRAFT/(name+'.png'))
    finally:studio.close()
    print(f'Five design previews ready: {DRAFT}',flush=True)


def prepare():
    assert not OUT.exists(),'Refusing to overwrite video output'
    study.verify_sources();cases=load_cases();media=media_helper()
    tests=(ROOT/'work/rr_severity_video_tests.log').read_text(encoding='utf-8-sig')
    assert '\nOK' in tests and 'FAILED' not in tests
    protected=dict(s.read(study.OUT/'design.json')['prior_artifacts_sha256'])
    for path in [*study.OUT.rglob('*'),*source_files(ROOT)]:
        if path.is_file():protected[str(path.relative_to(ROOT))]=d.digest(path)
    for name,sha in protected.items():assert d.digest(project_file(ROOT, name))==sha,name
    OUT.mkdir()
    with (OUT/'tests-before.log').open('x',encoding='utf-8') as handle:handle.write(tests)
    external={str(previous.HELPER):s.digest(previous.HELPER)}
    for name in ['lmroman10-regular.otf','lmroman10-italic.otf']:external[str(media.FONT_DIR/name)]=s.digest(media.FONT_DIR/name)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),pose=23000,size=SIZE,fps=FPS,silent=True,
        chapters=CHAPTERS,total_seconds=60,total_frames=1500,source_sha256=s.digest(__file__),
        test_sha256=s.digest(ROOT/'tests/test_make_rr_severity_video.py'),helper_sha256=s.digest(ROOT/'research/make_multimotor_video.py'),
        tests_before_sha256=s.digest(OUT/'tests-before.log'),
        external_sha256=external,protected_inputs_sha256=protected,
        selection='User-selected pose23000, all five conditions. Ten archived-control episodes and 15 prior severity episodes supply the summary; no new trials.',
        cases={name:dict(trace=c['row']['trace'],trace_sha256=c['row']['trace_sha256'],historical_control=c['row']['historical_control'],
            strength_remaining=c['row']['strength_remaining']) for name,c in cases.items()},
        method='Read qpos/qvel directly from saved NPZ files. Set rendering-model state then call mj_forward only. No integration, policy inference, training or evaluation. Tracking cameras share distance, azimuth and elevation.',
        design='Existing white/Latin Modern Roman video style; paired comparison, four-panel comparison, late-arrival timeline, all-five-pose table. Humanizer pass keeps factual English captions.',
        simulation=False,training=False,final_confirmation_seeds_used=False)
    s.write(OUT/'video_plan.json',plan)
    motion={}
    for name,c in cases.items():motion[name+'_qpos']=c['qpos'];motion[name+'_qvel']=c['qvel']
    with (OUT/'motion.npz').open('xb') as handle:np.savez_compressed(handle,**motion)
    s.write(OUT/'motion_verification.json',dict(physics_states=50005,qpos_qvel_copied_exactly=True,all_initial_states_paired=True,
        per_case={name:dict(states=len(c['qpos']),max_qpos_error=0.,max_qvel_error=0.,trace_sha256=c['row']['trace_sha256']) for name,c in cases.items()},
        source='Direct saved states, no replay integration. Rendering display state is checked on every rendered pose.'))
    storyboard='''# RR severity comparison storyboard

White background and Latin Modern Roman match the previous project videos. The existing Go2 model, amber knee highlight and green healthy labels carry through. Labels are factual English.

- 0 to 10 s: the unchanged mission criteria, shared pose23000, fault onset and archived-control disclosure.
- 10 to 30 s: healthy archived control beside the RR75 saved run, synchronized at1x. Tracking cameras keep the feet readable; progress rails show displacement from the initial heading.
- 30 to 40 s: RR75/50/25/10 synchronized at2x. RR10 remains visibly labelled as an archived control.
- 40 to 48 s: RR75 from simulation16 to20 s at1x, then a four-second freeze at20 s. Goal entry18.83s and final hold0.77s are values for pose23000.
- 48 to 60 s: all-five-pose success/fall counts, median final error and median hold. Healthy and RR10 rows are archive controls. New conditions all have0/5 mission success.

Caption pass: no unsupported causal claim, no implied success from entering the position tolerance, no claim that the displayed pose is the five-pose result. Archive sources stay labelled; the target fault10% remains a diagnostic failure.
'''
    with (OUT/'storyboard.md').open('x',encoding='utf-8') as handle:handle.write(storyboard)
    print('Frozen 60-second trace-only film; 50005 saved states; no new runs.',flush=True)


def verify_plan():
    p=s.read(OUT/'video_plan.json')
    assert s.digest(__file__)==p['source_sha256'] and s.digest(ROOT/'tests/test_make_rr_severity_video.py')==p['test_sha256']
    assert s.digest(OUT/'tests-before.log')==p['tests_before_sha256']
    assert s.digest(ROOT/'research/make_multimotor_video.py')==p['helper_sha256']
    for name,sha in p['external_sha256'].items():assert s.digest(name)==sha,name
    for name,sha in p['protected_inputs_sha256'].items():assert d.digest(project_file(ROOT, name))==sha,name
    return p


def render():
    p=verify_plan();assert not (OUT/'rr_severity_comparison.mp4').exists()
    media=media_helper();cases=load_cases();studio=studio_for(media)
    with np.load(OUT/'motion.npz',allow_pickle=False) as motion:
        for name,c in cases.items():
            np.testing.assert_array_equal(motion[name+'_qpos'],c['qpos']);np.testing.assert_array_equal(motion[name+'_qvel'],c['qvel'])
    path=OUT/'rr_severity_comparison.mp4'
    writer=media.imageio_ffmpeg.write_frames(str(path),SIZE,fps=FPS,codec='libx264',quality=7,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-an','-movflags','+faststart','-threads','2'])
    writer.send(None);frames=0
    try:
        opening_page=opening(media,studio,cases);ending=closing(media,cases)
        opening_page.save(OUT/'intro.png');ending.save(OUT/'summary.png')
        for chapter,seconds in CHAPTERS:
            for frame in range(seconds*FPS):
                if chapter=='mission':image=opening_page
                elif chapter=='paired':image=paired(media,studio,cases,sample_tick(frame))
                elif chapter=='levels':image=levels(media,studio,cases,sample_tick(frame,rate=2))
                elif chapter=='late_arrival':image=late_arrival(media,studio,cases,sample_tick(frame,start_s=16))
                else:image=ending
                writer.send(np.asarray(image));frames+=1
                if frame in [0,seconds*FPS-1]:image.save(OUT/f'{chapter}_{frame:03}.png')
                if frame%100==0:print(f'{chapter}: {frame}/{seconds*FPS}',flush=True)
    finally:writer.close();studio.close()
    assert frames==p['total_frames']
    s.write(OUT/'render_verification.json',dict(frames=frames,rendered_poses=len(studio.drawn),
        rendered_qpos_qvel_match_source_exactly=True,integration_steps=0,policy_predictions=0,
        motion_sha256=s.digest(OUT/'motion.npz'),video_sha256=s.digest(path)))
    verify()


def verify():
    p=verify_plan();media=media_helper();path=OUT/'rr_severity_comparison.mp4'
    count,seconds=media.imageio_ffmpeg.count_frames_and_secs(str(path));assert count==1500 and abs(seconds-60)<.01
    reader=media.imageio_ffmpeg.read_frames(str(path));meta=next(reader);reader.close()
    assert tuple(meta['size'])==SIZE and meta['fps']==FPS
    exe=media.imageio_ffmpeg.get_ffmpeg_exe()
    info=subprocess.run([exe,'-hide_banner','-i',str(path)],capture_output=True,text=True)
    assert 'Audio:' not in info.stderr
    subprocess.run([exe,'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,capture_output=True)
    sheet=Image.new('RGB',SIZE,'white')
    for i,t in enumerate([2,14,29.96,33,39.96,43,47.96,50,59.96]):
        target=OUT/f'decoded_{i}.png';assert not target.exists()
        subprocess.run([exe,'-v','error','-ss',str(t),'-i',str(path),'-frames:v','1',str(target)],check=True,capture_output=True)
        with Image.open(target) as im:
            assert np.asarray(im).std()>5
            sheet.paste(im.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(OUT/'contact_sheet.png')
    verify_plan()
    s.write(OUT/'verification.json',dict(passed=True,frames=count,seconds=seconds,size=meta['size'],fps=meta['fps'],
        silent=True,all_frames_decoded=True,saved_motion_states=50005,rendered_motion_matches_trace=True,
        archived_controls_labelled=True,summary_covers_five_poses=True,old_artifacts_unchanged=True,
        new_rollouts=0,training=False,video_sha256=s.digest(path)))
    with (OUT/'README.md').open('x',encoding='utf-8') as handle:
        handle.write('# RR severity video\n\nA 60-second silent film at 1920x1080 and 25 fps, showing saved pose 23000 under five conditions: healthy and RR at 75%, 50%, 25% and 10%. Motion comes directly from the saved joint poses. The main comparison plays at normal speed; the four-level comparison plays at 2x. The last frame in the RR 75% detail is held for reading.\n\nHealthy and RR 10% are archived controls. The closing table covers all five poses per condition. RR 75% enters the goal area too late for the final 2 s hold, so it fails the mission. No new simulation integration, policy inference, evaluation or training was run.\n\nThe existing white background, Latin Modern Roman typography and renderer are reused. Rendering changes only copied model colors and the tracking camera. See video_plan.json, storyboard.md, motion_verification.json, render_verification.json, verification.json and contact_sheet.png for checks. This is a flat-floor simulation; hardware and generalization remain untested.\n')
    print(f'Verified video ready: {path}',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['preview','prepare','render'])
    {'preview':preview,'prepare':prepare,'render':render}[parser.parse_args().stage]()
