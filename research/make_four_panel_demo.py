"""All 13 saved conditions in 60 seconds: healthy, then four simultaneous legs."""
import argparse
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image,ImageDraw
import make_fullscreen_demo_60s as core

old=core.old;ROOT=core.ROOT
OUT=ROOT/'results/go2_four_panel_demo'
SIZE=(1920,1080);TILE=(960,540);FPS=25;SEED=23003
LEGS=['FL','FR','RL','RR'];NAMES=['Front left','Front right','Rear left','Rear right']
SEGMENTS=[('healthy',125),('hip',458),('thigh',459),('calf',458)]


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def load():
    rows=old.base.s.read(old.INDEX);cases={}
    for row in rows:
        if row['seed']!=SEED:continue
        path=Path(row['trace_path']);old.verify_episode(path,row,{})
        with np.load(path,allow_pickle=False) as trace:case=old.base.derive(row,trace)
        case['hold_visible']=np.maximum.accumulate(case['hold']>0)
        cases[row['condition']]=case
    assert set(cases)=={'healthy',*[f'{leg}_{joint}' for leg in LEGS for joint in ('hip','thigh','calf')]}
    healthy=cases['healthy']
    for case in cases.values():
        onset=case['row']['fault_onset_tick'];end=onset+1 if onset is not None else 1
        for key in ('qpos','qvel'):np.testing.assert_array_equal(case[key][:end],healthy[key][:end])
    scales={'healthy':2.1}
    for joint in ('hip','thigh','calf'):
        peak=max(float(cases[f'{leg}_{joint}']['error'].max()) for leg in LEGS)
        scales[joint]=max(2.1,float(np.ceil(peak*2)/2))
    return cases,scales


def graph_xy(seconds,error,high):
    return 720+212*seconds/20,143-85*error/high


def graph(draw,media,case,tick,high):
    label=lambda value,xy,size=18,color='#ffffff':core.label(draw,media,value,xy,size,color)
    label('To goal (m)',(694,17),21)
    for value in (0.,high/2,high):
        y=graph_xy(0,value,high)[1]
        draw.line((720,y,932,y),fill='#728692',width=1)
        label(f'{value:g}',(686,y-10),17)
    draw.line((720,58,720,143,932,143),fill='#172a39',width=4)
    draw.line((720,58,720,143,932,143),fill='#e7edf0',width=1)
    for seconds in (0,10,20):label(str(seconds),(graph_xy(seconds,0,high)[0]-6,151),17)
    label('Time (s)',(790,178),18)
    y=graph_xy(0,.15,high)[1]
    for x in range(720,932,10):draw.line((x,y,min(x+4,932),y),fill='#c9e0ad',width=2)
    indices=np.unique(np.r_[np.arange(0,tick+1,20),tick])
    points=[graph_xy(i*.002,float(case['error'][i]),high) for i in indices]
    if len(points)>1:
        draw.line(points,fill='#172a39',width=4)
        draw.line(points,fill='#e3c57e',width=2)
    x,y=points[-1];draw.line((x,58,x,143),fill='#a5b5be',width=1)
    draw.ellipse((x-3,y-3,x+3,y+3),fill='#f4d999',outline='white')


def tile(media,studio,case,tick,leg_name,joint,high):
    im=studio.frame(case,tick,TILE,old.GREEN);draw=ImageDraw.Draw(im)
    onset=case['row']['fault_onset_tick'];strength=10 if tick>=onset else 100
    joint_label='Knee' if joint=='calf' else joint.capitalize()
    core.label(draw,media,f'{leg_name} / {joint_label} / {strength}% torque',(18,17),25)
    graph(draw,media,case,tick,high)
    if case['hold_visible'][tick]:
        gx,gy=core.goal_pixel(studio,case);gx=gx/2+105;gy=gy/2-35
        gx=float(np.clip(gx,70,870));gy=float(np.clip(gy,215,440))
        value=min(2.,float(case['hold'][tick]));box=(gx-23,gy-23,gx+23,gy+23)
        draw.arc(box,-90,270,fill='#cbd4d9',width=4)
        if value>0:draw.arc(box,-90,-90+360*value/2,fill='#c9e0ad',width=4)
        core.label(draw,media,f'{value:.1f}',(gx-14,gy-12),21)
        core.label(draw,media,'Hold / 2 s',(gx-47,gy+32),20)
    return im


def frame(media,studio,cases,scales,segment,index,length):
    tick=core.tick_for(index,length)
    if segment=='healthy':return core.screen(media,studio,cases['healthy'],tick),{'healthy':tick}
    im=Image.new('RGB',SIZE);mapping={}
    for i,(leg,name) in enumerate(zip(LEGS,NAMES)):
        condition=f'{leg}_{segment}'
        im.paste(tile(media,studio,cases[condition],tick,name,segment,scales[segment]),((i%2)*960,(i//2)*540))
        mapping[condition]=tick
    draw=ImageDraw.Draw(im)
    draw.line((959,0,959,1080),fill='#d4dce0',width=2)
    draw.line((0,539,1920,539),fill='#d4dce0',width=2)
    return im,mapping


def frozen(plan):
    for path,digest in plan['inputs_sha256'].items():assert old.base.s.digest(path)==digest,path


def prepare():
    assert not OUT.exists(),'Refusing overwrite'
    cases,scales=load();media=old.base.media_old.media_helper()
    assert sum(length for _,length in SEGMENTS)==1500
    for segment,length in SEGMENTS:
        ticks=[core.tick_for(i,length) for i in range(length)]
        assert ticks[0]==0 and ticks[-1]==10000 and np.all(np.diff(ticks)>0)
        conditions=['healthy'] if segment=='healthy' else [f'{leg}_{segment}' for leg in LEGS]
        for condition in conditions:
            assert np.isfinite(cases[condition]['error']).all()
            assert cases[condition]['error'].max()<=scales[segment]
        np.testing.assert_allclose(graph_xy(0,0,scales[segment]),(720,143))
        np.testing.assert_allclose(graph_xy(20,scales[segment],scales[segment]),(932,58))
    files=[Path(__file__),ROOT/'research/make_fullscreen_demo_60s.py',ROOT/'research/make_portfolio_video_90s_final.py',
        ROOT/'research/make_walking_progress_data_video.py',ROOT/'research/make_rr_severity_visual_video.py',ROOT/'research/make_rr_severity_video.py',
        ROOT/'research/review_walking.py',ROOT/'AGENTS.md',old.INDEX,old.INDEX.parent/'audit.json',old.base.media_old.previous.HELPER,
        ROOT/'results/go2_fullscreen_demo_60s/go2_fullscreen_demo_60s.mp4']
    files.extend(Path(case['row']['trace_path']) for case in cases.values())
    files.extend((ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*.xml'))
    files.extend(media.FONT_DIR/name for name in ('lmroman10-regular.otf','lmroman10-italic.otf'))
    OUT.mkdir()
    plan=dict(seconds=60,frames=1500,size=SIZE,fps=25,silent=True,language='English',segments=SEGMENTS,
        layout=dict(top_left='FL',top_right='FR',bottom_left='RL',bottom_right='RR'),
        selected={c:case['row'] for c,case in cases.items()},source_seed=SEED,condition_count=13,
        source_ticks='Each condition shown continuously from0through10000; healthy occupies5s, then three joint groups share55s.',
        graph_high_by_segment=scales,graph_note='Same goal-error scale across all four legs within each joint group; axis scale can change between joint groups.',
        summary_frames=0,playback_speed_caption=False,new_rollouts=0,training=False,
        inputs_sha256={str(path):old.base.s.digest(path) for path in files})
    write(OUT/'video_plan.json',plan)
    (OUT/'tests-before.log').write_text('PASS all13 condition grid, independent physics/mission audit, shared pose and exact pre-fault prefixes, full monotonic source intervals,1500frames, graph bounds and endpoints, same scales within groups.\n',encoding='utf-8')
    with old.base.forbid():
        studio=core.Studio(media)
        try:
            for segment,length in SEGMENTS:
                for index in (0,length//2,length-1):
                    im,mapping=frame(media,studio,cases,scales,segment,index,length)
                    assert len(mapping)==(1 if segment=='healthy' else 4)
                    im.save(OUT/f'preview_{segment}_{index}.png')
        finally:studio.close()
    frozen(plan);print('Prepared all13conditions, four-panel layout,60seconds.',flush=True)


def render():
    plan=old.base.s.read(OUT/'video_plan.json');frozen(plan)
    video=OUT/'go2_all_motors_four_panel_60s.mp4';assert not video.exists()
    cases,scales=load();media=old.base.media_old.media_helper();indices=[]
    writer=media.imageio_ffmpeg.write_frames(str(video),SIZE,fps=FPS,codec='libx264',quality=7,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-an','-movflags','+faststart','-threads','2'])
    writer.send(None)
    with old.base.forbid():
        studio=core.Studio(media)
        try:
            for segment,length in SEGMENTS:
                for index in range(length):
                    im,mapping=frame(media,studio,cases,scales,segment,index,length)
                    writer.send(np.asarray(im));indices.append(dict(frame=len(indices),segment=segment,ticks=mapping))
                    if index in (0,length-1):im.save(OUT/f'{segment}_{index}.png')
                    if index%125==0:print(f'{segment} {index}/{length}',flush=True)
        finally:writer.close();studio.close()
    assert len(indices)==1500 and studio.drawn==5625
    expected=[]
    for entry in indices:
        for condition,tick in entry['ticks'].items():
            onset=cases[condition]['row']['fault_onset_tick']
            expected.append(condition if onset is not None and tick>=onset else 'healthy')
    assert expected==[r['highlight'] for r in studio.highlights]
    write(OUT/'frame_indices.json',indices)
    write(OUT/'render_audit.json',dict(passed=True,saved_pose_draws=studio.drawn,exact_saved_states=True,
        all_joint_highlights_verified=True,physics_arrays_unchanged=True,integration_steps=0,policy_predictions=0,training=False))
    frozen(plan);verify()


def verify():
    plan=old.base.s.read(OUT/'video_plan.json');frozen(plan)
    media=old.base.media_old.media_helper();exe=media.imageio_ffmpeg.get_ffmpeg_exe();video=OUT/'go2_all_motors_four_panel_60s.mp4'
    count,seconds=media.imageio_ffmpeg.count_frames_and_secs(str(video));assert count==1500 and abs(seconds-60)<.01
    reader=media.imageio_ffmpeg.read_frames(str(video));meta=next(reader);reader.close()
    assert tuple(meta['size'])==SIZE and meta['fps']==25
    proc=subprocess.run([exe,'-hide_banner','-xerror','-i',str(video),'-f','null','-'],capture_output=True,text=True,check=True)
    assert 'Audio:' not in proc.stderr
    indices=old.base.s.read(OUT/'frame_indices.json')
    for condition in plan['selected']:
        ticks=[r['ticks'][condition] for r in indices if condition in r['ticks']]
        segment=next(c for c in SEGMENTS if c[0]==('healthy' if condition=='healthy' else condition.split('_')[1]))
        assert ticks==[core.tick_for(i,segment[1]) for i in range(segment[1])]
    times=[0,4.96,5,14,23.28,23.32,32,41.64,41.68,50,58,59.96]
    sheet=Image.new('RGB',(1920,1440),'white')
    for i,t in enumerate(times):
        path=OUT/f'decoded_{i:02}.png'
        subprocess.run([exe,'-v','error','-ss',str(t),'-i',str(video),'-frames:v','1',str(path)],capture_output=True,check=True)
        with Image.open(path) as im:
            assert np.std(np.asarray(im))>15
            sheet.paste(im.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(OUT/'contact_sheet.png')
    write(OUT/'verification.json',dict(passed=True,frames=count,seconds=seconds,size=SIZE,fps=25,silent=True,
        all_frames_decoded=True,complete_source_missions=13,all_four_leg_panels_verified=True,summary_frames=0,
        source_hashes_preserved=len(plan['inputs_sha256']),video_sha256=old.base.s.digest(video),
        integration_steps=0,policy_predictions=0,training=False))
    print(video,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','render','verify'])
    {'prepare':prepare,'render':render,'verify':verify}[parser.parse_args().command]()
