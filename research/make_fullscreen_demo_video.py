"""Four uninterrupted saved missions, fullscreen arena and minimal English overlays."""
import argparse
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import make_portfolio_video_90s_final as old

ROOT=old.ROOT
OUT=ROOT/'results/go2_fullscreen_demo'
SIZE=(1920,1080);FPS=25
ORDER=['healthy','RR_calf','RL_thigh','FL_calf']
LABELS={'healthy':'Healthy','RR_calf':'Rear-right knee','RL_thigh':'Rear-left thigh','FL_calf':'Front-left knee'}
LENGTHS=[563,562,563,562]
SEED=23003


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def tick_for(frame,length):
    return round(frame*10000/(length-1))


def load():
    rows=old.base.s.read(old.INDEX);cases={}
    for condition in ORDER:
        row=next(r for r in rows if r['condition']==condition and r['seed']==SEED)
        path=Path(row['trace_path'])
        old.verify_episode(path,row,{})
        with np.load(path,allow_pickle=False) as a:case=old.base.derive(row,a)
        case['hold_visible']=np.maximum.accumulate(case['hold']>0)
        cases[condition]=case
    reference=cases['healthy']
    for condition,case in cases.items():
        onset=case['row']['fault_onset_tick']
        cutoff=onset+1 if onset is not None else 1
        for name in ('qpos','qvel'):
            np.testing.assert_array_equal(case[name][:cutoff],reference[name][:cutoff])
    assert [cases[c]['row']['success'] for c in ORDER]==[True,True,True,False]
    return cases


def label(draw,media,value,xy,size=31,fill='#ffffff'):
    draw.text(xy,value,font=media.font(size),fill=fill,stroke_width=2,stroke_fill='#172a39')


def goal_pixel(studio,case):
    scene=studio.base.renderer.scene
    # Average the two GL cameras, even when rendering in mono.
    position=np.mean([cam.pos for cam in scene.camera],axis=0)
    forward=np.mean([cam.forward for cam in scene.camera],axis=0);forward/=np.linalg.norm(forward)
    up=np.mean([cam.up for cam in scene.camera],axis=0);up/=np.linalg.norm(up)
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    yaw=case['row']['initial_yaw']
    world=np.r_[np.asarray(case['row']['initial_xy'])+2*np.array([np.cos(yaw),np.sin(yaw)]),.015]
    delta=world-position;depth=float(delta@forward)
    assert depth>0
    cam=scene.camera[0]
    half_height=(cam.frustum_top-cam.frustum_bottom)/2
    center_height=(cam.frustum_top+cam.frustum_bottom)/2
    y_plane=(delta@up)*cam.frustum_near/depth
    x_plane=(delta@right)*cam.frustum_near/depth
    half_width=half_height*SIZE[0]/SIZE[1]
    return ((x_plane/(2*half_width)+.5)*SIZE[0],(.5-(y_plane-center_height)/(2*half_height))*SIZE[1])


def screen(media,studio,case,tick):
    row=case['row'];condition=row['condition'];onset=row['fault_onset_tick']
    image=studio.frame(case,tick,SIZE,old.GREEN)
    draw=ImageDraw.Draw(image)
    active=onset is not None and tick>=onset
    title=LABELS[condition]
    if condition!='healthy':title+=f'  /  {10 if active else 100}% torque'
    label(draw,media,title,(42,32),35)
    label(draw,media,f'Time {tick*.002:04.1f} / 20 s',(1550,32),30)
    label(draw,media,f"To goal {case['error'][tick]:.2f} m",(1550,78),30)
    label(draw,media,'0.89x playback',(42,1012),25,fill='#e1e6e8')
    if case['hold_visible'][tick]:
        gx,gy=goal_pixel(studio,case)
        gx=float(np.clip(gx,125,SIZE[0]-160));gy=float(np.clip(gy-125,210,SIZE[1]-160))
        value=min(2.,float(case['hold'][tick]));box=(gx-34,gy-34,gx+34,gy+34)
        draw.arc(box,-90,270,fill='#cbd4d9',width=5)
        if value>0:draw.arc(box,-90,-90+360*value/2,fill='#c9e0ad',width=6)
        label(draw,media,f'{value:.1f}',(gx-18,gy-15),27)
        label(draw,media,'Hold / 2 s',(gx-65,gy+44),26)
    return image


def frozen(plan):
    for name,digest in plan['inputs_sha256'].items():assert old.base.s.digest(name)==digest,name


def prepare():
    assert not OUT.exists(),'Refusing overwrite'
    cases=load();media=old.base.media_old.media_helper()
    assert sum(LENGTHS)==2250
    for length in LENGTHS:
        ticks=[tick_for(i,length) for i in range(length)]
        assert ticks[0]==0 and ticks[-1]==10000 and np.all(np.diff(ticks)>0)
        assert max(abs(np.diff(ticks)-10000/(length-1)))<1.1
    for case in cases.values():
        q=case['qpos'];v=case['qvel'];row=case['row'];w,x,y,z=q[:,3:7].T
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        heading=np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
        stable=(case['error']<=.15)&(heading<=np.pi/12)&(np.linalg.norm(v[:,:2],axis=1)<=.05)
        assert np.all(case['hold'][~stable]==0)
        for at in np.flatnonzero(case['hold']>0):
            assert stable[at] and case['hold_visible'][at]
        assert case['hold_visible'][0]==False
    files=[Path(__file__),ROOT/'research/make_portfolio_video_90s_final.py',ROOT/'research/make_walking_progress_data_video.py',
        ROOT/'research/make_rr_severity_visual_video.py',ROOT/'research/make_rr_severity_video.py',ROOT/'research/review_walking.py',old.INDEX,
        old.INDEX.parent/'audit.json',ROOT/'AGENTS.md',old.base.media_old.previous.HELPER,
        ROOT/'results/go2_portfolio_video_90s_final/go2_rebalance_portfolio_90s.mp4']
    files.extend(Path(case['row']['trace_path']) for case in cases.values())
    files.extend((ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*.xml'))
    files.extend(media.FONT_DIR/name for name in ('lmroman10-regular.otf','lmroman10-italic.otf'))
    OUT.mkdir()
    plan=dict(seconds=90,frames=2250,size=SIZE,fps=25,silent=True,language='English',fullscreen=True,
        order=ORDER,frame_lengths=LENGTHS,source_seed=SEED,selected={c:case['row'] for c,case in cases.items()},
        playback='Each full 20-second mission occupies about22.5s, uniform nominal0.89x; 3 cuts only. No pauses, summaries or intro.',
        data_overlays='Saved path; world lane +/-0.20m; goal radius0.15m; actual weakened joint; time; 2D goal error; true stable hold near target.',
        limitations='Four selected development examples, not all motor conditions. Three success examples and one failure; see README for complete context.',
        inputs_sha256={str(p):old.base.s.digest(p) for p in files},integration_steps=0,policy_predictions=0,training=False)
    write(OUT/'video_plan.json',plan)
    (OUT/'tests-before.log').write_text('PASS full-episode monotonic sampling/endpoints, 2250 frames/90seconds, initial-state and pre-fault equality, source hashes, independent physics/mission audit, hold position/heading/speed boundaries.\n',encoding='utf-8')
    with old.base.forbid():
        studio=old.Studio(media)
        try:
            for condition,case in cases.items():
                for tick in [0,case['row']['fault_onset_tick'] or 0,8000,10000]:
                    image=screen(media,studio,case,tick)
                    px,py=goal_pixel(studio,case)
                    assert 0<px<SIZE[0] and 0<py<SIZE[1]
                    image.save(OUT/f'preview_{condition}_{tick}.png')
        finally:studio.close()
    frozen(plan)
    print('Prepared four complete missions, fullscreen,90s,English,no summaries.',flush=True)


def render():
    plan=old.base.s.read(OUT/'video_plan.json');frozen(plan)
    target=OUT/'go2_fullscreen_demo_90s.mp4';assert not target.exists()
    cases=load();media=old.base.media_old.media_helper();indices=[]
    writer=media.imageio_ffmpeg.write_frames(str(target),SIZE,fps=FPS,codec='libx264',quality=7,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-an','-movflags','+faststart','-threads','2'])
    writer.send(None)
    with old.base.forbid():
        studio=old.Studio(media)
        try:
            for condition,length in zip(ORDER,LENGTHS):
                case=cases[condition]
                for frame in range(length):
                    tick=tick_for(frame,length);im=screen(media,studio,case,tick)
                    writer.send(np.asarray(im));indices.append(dict(frame=len(indices),condition=condition,tick=tick))
                    if frame in (0,length-1):im.save(OUT/f'{condition}_{frame}.png')
                    if frame%125==0:print(f'{condition} {frame}/{length}',flush=True)
        finally:writer.close();studio.close()
    assert len(indices)==2250
    write(OUT/'frame_indices.json',indices)
    assert len(studio.highlights)==2250
    for row,highlight in zip(indices,studio.highlights):
        case=cases[row['condition']];onset=case['row']['fault_onset_tick']
        expected=row['condition'] if onset is not None and row['tick']>=onset else 'healthy'
        assert highlight['highlight']==expected
    write(OUT/'render_audit.json',dict(passed=True,exact_saved_states_every_draw=True,physics_arrays_unchanged=True,
        all_highlights_verified=True,draws=studio.drawn,integration_steps=0,policy_predictions=0,training=False))
    frozen(plan);verify()


def verify():
    plan=old.base.s.read(OUT/'video_plan.json');frozen(plan)
    media=old.base.media_old.media_helper();exe=media.imageio_ffmpeg.get_ffmpeg_exe();video=OUT/'go2_fullscreen_demo_90s.mp4'
    count,seconds=media.imageio_ffmpeg.count_frames_and_secs(str(video));assert count==2250 and abs(seconds-90)<.01
    reader=media.imageio_ffmpeg.read_frames(str(video));meta=next(reader);reader.close()
    assert tuple(meta['size'])==SIZE and meta['fps']==25
    proc=subprocess.run([exe,'-hide_banner','-xerror','-i',str(video),'-f','null','-'],capture_output=True,text=True,check=True)
    assert 'Audio:' not in proc.stderr
    indices=old.base.s.read(OUT/'frame_indices.json')
    for condition,length in zip(ORDER,LENGTHS):
        ticks=[r['tick'] for r in indices if r['condition']==condition]
        assert len(ticks)==length and ticks==[tick_for(i,length) for i in range(length)]
    times=[0,10,22.48,22.52,34,44.96,45,56,67.48,67.52,79,89.96]
    sheet=Image.new('RGB',(1920,1440),'white')
    for i,t in enumerate(times):
        path=OUT/f'decoded_{i:02}.png'
        subprocess.run([exe,'-v','error','-ss',str(t),'-i',str(video),'-frames:v','1',str(path)],capture_output=True,check=True)
        with Image.open(path) as image:
            assert np.std(np.asarray(image))>15
            sheet.paste(image.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(OUT/'contact_sheet.png')
    write(OUT/'verification.json',dict(passed=True,frames=count,seconds=seconds,size=SIZE,fps=25,silent=True,
        all_frames_decoded=True,complete_missions=4,scene_cuts=3,summary_frames=0,source_hashes_preserved=len(plan['inputs_sha256']),
        video_sha256=old.base.s.digest(video),integration_steps=0,policy_predictions=0,training=False))
    print(video,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','render','verify'])
    {'prepare':prepare,'render':render,'verify':verify}[parser.parse_args().command]()
