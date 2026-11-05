"""Render the approved silent English portfolio film from saved states only."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
from unittest.mock import patch
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import make_walking_progress_data_video as base
import make_rr_severity_visual_video as visual
from review_walking import verify_episode

ROOT=Path(__file__).resolve().parents[1]
C=Path('C:/Users/Zidan/Documents/Codex')
OUT=ROOT/'results/go2_portfolio_video_90s_final'
SIZE=(1920,1080);FPS=25
CHAPTERS=[('mission',10),('fault',10),('paired',20),('hold',15),('failures',15),('stop',10),('results',10)]
JOINTS=[f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
INK=base.INK;MUTED=base.MUTED;GREEN='#536c43';AMBER='#a35317';GRID='#e3e5e1'
INDEX=C/'go2-all12-rr-hip-pitch175/episodes.json'


def text(media,im,value,x,y,size=28,color=INK):media.put(im,value,x,y,size,fill=color)
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def selected_rows():
    rows=base.s.read(INDEX)
    assert len(rows)==65 and len({(r['condition'],r['seed']) for r in rows})==65
    passing=[r for r in rows if r['condition']=='RR_calf' and r['success']]
    middle=np.median([r['final_goal_error_m'] for r in passing])
    rr=min(passing,key=lambda r:(round(abs(r['final_goal_error_m']-middle),12),r['seed']))
    selected={'rear':rr,'healthy':next(r for r in rows if r['condition']=='healthy' and r['seed']==rr['seed'])}
    for name,condition in [('hip','FL_hip'),('knee','FL_calf')]:
        group=[r for r in rows if r['condition']==condition]
        middle=np.median([r['final_progress_m'] for r in group])
        selected[name]=min(group,key=lambda r:(abs(r['final_progress_m']-middle),r['seed']))
    for name,folder in [('radius12',C/'go2-all12-fl-hip-cross-feature'),('radius10',ROOT/'results/walking_fl_hip_stop10_20261105')]:
        group=[r for r in base.s.read(folder/'episodes.json') if r['condition']=='FL_hip']
        # This pose loses success in the intervention; all five outcomes remain on screen.
        selected[name]=next(r for r in group if r['seed']==23003)
    return rows,selected


def load():
    rows,selected=selected_rows();cases={}
    for name,row in selected.items():
        path=Path(row['trace_path']);assert base.s.digest(path)==row['trace_sha256']
        verify_episode(path,row,{})
        with np.load(path,allow_pickle=False) as a:cases[name]=base.derive(row,a)
    assert cases['rear']['row']['success'] and cases['healthy']['row']['success']
    np.testing.assert_array_equal(cases['healthy']['qpos'][0],cases['rear']['qpos'][0])
    np.testing.assert_array_equal(cases['healthy']['qvel'][0],cases['rear']['qvel'][0])
    return rows,cases


def frame_ticks(chapter,frame,cases):
    if chapter=='mission':return {'healthy':min(10000,frame*40)}
    if chapter=='fault':return {'rear':min(10000,cases['rear']['row']['fault_onset_tick']-1000+frame*20)}
    if chapter=='paired':return {'healthy':frame*20,'rear':frame*20}
    if chapter=='hold':return {'healthy':6250+frame*10,'rear':6250+frame*10}
    if chapter=='failures':return {'hip':2500+frame*20,'knee':2500+frame*20}
    if chapter=='stop':return {'radius12':7500+frame*10,'radius10':7500+frame*10}
    return {'rear':10000}


class Studio(visual.VisualStudio):
    def __init__(self,media):
        super().__init__(media)
        self.base.model.vis.quality.shadowsize=4096
        self.highlights=[]

    def frame(self,case,tick,size,color=GREEN,close=False):
        b=self.base;m=b.model;row=case['row']
        if self.size!=size:
            b.renderer.close();b.renderer=mujoco.Renderer(m,width=size[0],height=size[1]);self.size=size
        onset=row['fault_onset_tick'];active=onset is not None and tick>=onset
        condition=row['condition'] if active else 'healthy'
        for joint in JOINTS:
            mask=(m.geom_bodyid==m.body(joint).id)&(m.geom_group==2)
            assert mask.any(),joint
            m.geom_matid[mask]=b.original.geom_matid[mask]
            m.geom_rgba[mask]=b.original.geom_rgba[mask]
            if joint==condition:
                m.geom_matid[mask]=-1;m.geom_rgba[mask]=[.83,.58,.27,1.]
        b.data.qpos[:]=case['qpos'][tick];b.data.qvel[:]=case['qvel'][tick]
        mujoco.mj_forward(m,b.data)
        np.testing.assert_array_equal(b.data.qpos,case['qpos'][tick])
        np.testing.assert_array_equal(b.data.qvel,case['qvel'][tick])
        yaw=row['initial_yaw'];forward=np.array([np.cos(yaw),np.sin(yaw),0.])
        side=np.array([-forward[1],forward[0],0.]);start=np.r_[row['initial_xy'],.009];goal=start+2*forward
        current=b.data.qpos[:3].copy();current[2]=.18
        b.camera.azimuth=105;b.camera.elevation=-30;m.vis.global_.fovy=48
        if close:
            b.camera.lookat[:]=current;b.camera.distance=1.45
        else:
            b.camera.lookat[:]=(current+goal)/2+[0,0,.10]
            b.camera.distance=max(2.7,float(np.linalg.norm(current[:2]-goal[:2]))*1.35+1.)
        b.renderer.update_scene(b.data,camera=b.camera)
        b.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
        b.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        for value in np.arange(0,2,.15):
            for cross in (-.2,.2):self.line(start+value*forward+cross*side,start+min(value+.06,2)*forward+cross*side,[.6,.7,.75,1],.002)
        self.line(goal-.30*side,goal+.30*side,[.63,.82,.58,1],.006)
        for angle in np.linspace(0,2*np.pi,49)[:-1]:
            self.line(goal+.15*(np.cos(angle)*forward+np.sin(angle)*side),goal+.15*(np.cos(angle+np.pi/24)*forward+np.sin(angle+np.pi/24)*side),[.63,.82,.58,1],.004)
        rgba=tuple(int(color[i:i+2],16)/255 for i in (1,3,5))+(1.,)
        points=case['qpos'][np.unique(np.r_[np.arange(0,tick+1,100),tick]),:2]
        for a,z in zip(points[:-1],points[1:]):
            if np.linalg.norm(z-a)>1e-7:self.line(np.r_[a,.015],np.r_[z,.015],rgba,.004)
        if active:
            at=b.data.xpos[m.body(condition).id].copy()
            self.line(at,at+[0,0,.08],[.94,.61,.2,1],.008)
        self.highlights.append(dict(condition=row['condition'],tick=int(tick),highlight=condition))
        self.drawn+=1
        return Image.fromarray(b.renderer.render())


def graph(media,im,cases,now,names,field,box,title,colors):
    x,y,w,h=box;draw=ImageDraw.Draw(im);text(media,im,title,x,y,29)
    allvalues=np.concatenate([cases[n][field] for n in names])
    low=0. if field in ('error','planar_speed') else min(-.22,float(allvalues.min())-.04)
    high=max(.25 if field!='error' else 2.1,float(allvalues.max())+.04)
    left,top,right,bottom=x+62,y+46,x+w-15,y+h-36
    def xy(t,v):return left+(right-left)*t/20,bottom-(bottom-top)*(v-low)/(high-low)
    values=[0.,1.,2.] if field=='error' and high<2.3 else np.linspace(low,high,4)
    for value in values:
        draw.line((xy(0,value),xy(20,value)),fill=GRID,width=1)
        text(media,im,f'{value:.2f}' if abs(value)<1 else f'{value:.1f}',x,y+37+(bottom-top)*(high-value)/(high-low),20,MUTED)
    draw.line((left,top,left,bottom,right,bottom),fill='#aeb7ad',width=1)
    limit=.15 if field=='error' else .05 if field=='planar_speed' else .20
    for value in ([limit,-limit] if field=='lateral' else [limit]):
        if low<=value<=high:
            yy=xy(0,value)[1]
            for xx in np.arange(left,right,12):draw.line((xx,yy,min(xx+5,right),yy),fill=MUTED,width=1)
    for name in names:
        tick=now[name];indices=np.unique(np.r_[np.arange(0,tick+1,20),tick])
        points=[xy(i*.002,cases[name][field][i]) for i in indices]
        if len(points)>1:draw.line(points,fill=colors[name],width=3)
        px,py=points[-1];draw.ellipse((px-4,py-4,px+4,py+4),fill=colors[name])
    text(media,im,'0',left,bottom+8,20,MUTED);text(media,im,'20 s',right-45,bottom+8,20,MUTED)


def hold(media,im,case,tick,x,y,color):
    value=case['hold'][tick];draw=ImageDraw.Draw(im)
    draw.arc((x,y,x+105,y+105),-90,270,fill=GRID,width=8)
    if value>0:draw.arc((x,y,x+105,y+105),-90,-90+360*min(value/2,1),fill=color,width=8)
    text(media,im,f'{min(value,2):.1f} s',x+24,y+34,27,color)
    text(media,im,'Hold / 2 s',x+132,y+13,29)
    text(media,im,f"Goal error {case['error'][tick]:.3f} m",x+132,y+58,29)


def screen(media,studio,rows,cases,chapter,frame):
    im=Image.new('RGB',SIZE,'white');draw=ImageDraw.Draw(im);now=frame_ticks(chapter,frame,cases)
    if chapter=='mission':
        tick=now['healthy'];text(media,im,'Healthy',40,26,36,GREEN)
        im.paste(studio.frame(cases['healthy'],tick,(1240,690)),(40,90))
        base.space_time(media,im,{'healthy':cases['healthy']},{'healthy':tick},(1310,140,560,480))
        hold(media,im,cases['healthy'],tick,1350,690,GREEN)
        text(media,im,'2 m / 20 s',40,850,40)
        text(media,im,'Goal 0.15 m   /   heading 15°   /   path ±0.20 m',40,920,29,MUTED)
        text(media,im,'Hold 2 s   /   speed 0.05 m/s max',40,967,29,MUTED)
    elif chapter=='fault':
        tick=now['rear'];case=cases['rear'];active=tick>=case['row']['fault_onset_tick']
        text(media,im,'One motor at a time',40,26,36)
        im.paste(studio.frame(case,tick,(1240,740),AMBER,close=True),(40,100))
        text(media,im,'Rear-right knee',1340,190,34,AMBER)
        text(media,im,f'Torque {10 if active else 100}%',1340,265,40,AMBER)
        text(media,im,'Fault at 0.50 m',1340,375,30)
        text(media,im,f"Progress {case['progress'][tick]:.2f} m",1340,432,29,MUTED)
        text(media,im,'Other motors 100%',1340,540,30,MUTED)
        text(media,im,'Recorded excerpt',40,917,27,MUTED)
    elif chapter in ('paired','hold','failures','stop'):
        names={'paired':['healthy','rear'],'hold':['healthy','rear'],'failures':['hip','knee'],'stop':['radius12','radius10']}[chapter]
        labels={'healthy':'Healthy','rear':'Rear-right knee','hip':'Front-left hip','knee':'Front-left knee','radius12':'Stop radius 0.12 m','radius10':'Stop radius 0.10 m'}
        colors={names[0]:GREEN,names[1]:AMBER}
        for i,name in enumerate(names):
            case=cases[name];tick=now[name];x=40+i*960;color=colors[name]
            text(media,im,labels[name],x,26,35,color)
            im.paste(studio.frame(case,tick,(880,490),color,close=chapter in ('hold','stop')),(x,95))
            if chapter=='paired':
                fault=case['row']['fault_onset_tick'];strength=100 if fault is None or tick<fault else 10
                text(media,im,f'Torque {strength}%',x,605,28,MUTED)
                hold(media,im,case,tick,x+15,675,color)
                text(media,im,f"Progress {case['progress'][tick]:.2f} m / 2 m",x,865,32)
            elif chapter=='hold':
                hold(media,im,case,tick,x+15,645,color)
                text(media,im,f"Speed {case['planar_speed'][tick]:.3f} m/s",x,810,32)
                text(media,im,'Stop speed 0.05 m/s max',x,885,27,MUTED)
            elif chapter=='failures':
                graph(media,im,cases,now,[name],'lateral' if name=='hip' else 'error',(x,625,880,270),
                    'Lateral deviation (m)' if name=='hip' else 'To goal (m)',colors)
                text(media,im,'Path deviation' if name=='hip' else 'Slow progress',x,930,31,color)
                text(media,im,f"t = {tick*.002:05.2f} s",x+640,930,26,MUTED)
            else:
                hold(media,im,case,tick,x+10,630,color)
                text(media,im,'2/5 completed' if name=='radius12' else '1/5 completed',x,805,40,color)
                text(media,im,'5 starting poses',x,875,28,MUTED)
        if chapter=='stop':text(media,im,'Separate front-left hip test   /   Change not retained',40,972,29,MUTED)
    else:
        text(media,im,'Mission success',40,26,39)
        text(media,im,'8 / 12 conditions meet target',40,105,44,GREEN)
        im.paste(studio.frame(cases['rear'],10000,(620,430)),(40,220))
        text(media,im,'Healthy 5/5',40,730,38,GREEN)
        text(media,im,'Falls 0/65',40,805,35)
        text(media,im,'Target: at least 3/5 per motor',40,893,28,MUTED)
        x0=900;columns=[1080,1370,1660]
        for x,label in zip(columns,['Hip','Thigh','Knee']):text(media,im,label,x,215,31)
        for i,(leg,label) in enumerate([('FL','Front left'),('FR','Front right'),('RL','Rear left'),('RR','Rear right')]):
            y=300+i*145;text(media,im,label,x0-170,y+15,29)
            for x,joint in zip(columns,['hip','thigh','calf']):
                group=[r for r in rows if r['condition']==f'{leg}_{joint}'];assert len(group)==5
                count=sum(r['success'] for r in group);color=GREEN if count>=3 else AMBER
                for k,r in enumerate(sorted(group,key=lambda r:r['seed'])):
                    xx=x+k*28;draw.ellipse((xx,y,xx+14,y+14),fill=color if r['success'] else 'white',outline=color,width=2)
                text(media,im,f'{count}/5',x+30,y+38,34,color)
        text(media,im,'Development results   /   5 starting poses per motor',730,948,28,MUTED)
    rate={'mission':'2×','fault':'1×','paired':'1×','hold':'0.5×','failures':'1×','stop':'0.5×','results':''}[chapter]
    if chapter=='results':footer='Saved simulation   /   New-seed validation and hardware testing pending'
    elif chapter=='failures':footer='Recorded excerpts   /   '+rate
    else:
        tick=next(iter(now.values()));footer=f't = {tick*.002:05.2f} s   /   {rate}'+('   /   Recorded excerpt' if chapter in ('fault','hold','stop') else '')
    text(media,im,footer,40,1030,25,MUTED)
    return im


def frozen(plan):
    for name,digest in plan['inputs_sha256'].items():assert base.s.digest(name)==digest,name


def prepare():
    assert not OUT.exists(),'Refusing overwrite'
    rows,cases=load()
    assert sum(s for _,s in CHAPTERS)==90
    assert sum(r['success'] for r in rows)==32 and not any(r['fallen'] for r in rows)
    assert sum(sum(r['success'] for r in rows if r['condition']==c)>=3 for c in JOINTS)==8
    for chapter,seconds in CHAPTERS:
        for frame in (0,seconds*FPS-1):
            assert all(0<=t<=10000 for t in frame_ticks(chapter,frame,cases).values())
    assert frame_ticks('paired',100,cases)=={'healthy':2000,'rear':2000}
    for condition in JOINTS:
        assert condition in JOINTS
    files=[Path(__file__),INDEX,INDEX.parent/'audit.json',ROOT/'AGENTS.md',
        ROOT/'research/make_walking_progress_data_video.py',ROOT/'research/make_rr_severity_visual_video.py',
        ROOT/'research/make_rr_severity_video.py',ROOT/'research/make_multimotor_video.py',ROOT/'research/review_walking.py',
        base.media_old.previous.HELPER,ROOT/'vendor/mujoco_menagerie/unitree_go2/scene.xml']
    files.extend(Path(r['trace_path']) for r in rows)
    for folder in (C/'go2-all12-fl-hip-cross-feature',ROOT/'results/walking_fl_hip_stop10_20261105'):
        files.extend(folder/name for name in ('episodes.json','audit.json','protocol.json'))
        selected=[r for r in base.s.read(folder/'episodes.json') if r['condition']=='FL_hip']
        assert len(selected)==5
        for r in selected:
            verify_episode(Path(r['trace_path']),r,{})
            files.append(Path(r['trace_path']))
    for row in rows:assert base.s.digest(row['trace_path'])==row['trace_sha256']
    media=base.media_old.media_helper()
    files.extend(media.FONT_DIR/name for name in ('lmroman10-regular.otf','lmroman10-italic.otf'))
    OUT.mkdir()
    plan=dict(seconds=90,frames=2250,size=SIZE,fps=FPS,silent=True,language='English',chapters=CHAPTERS,
        no_control_diagram=True,selected={k:v['row'] for k,v in cases.items()},
        selection='Rear success nearest median successful error, matched healthy pose; front failures nearest median final progress. Stop-test pose23003 loses success, all five outcome counts shown.',
        results=dict(healthy=5,passing_motors=8,motors=12,completed=32,total=65,falls=0),
        inputs_sha256={str(f):base.s.digest(f) for f in files},new_rollouts=0,training=False,policy_predictions=0)
    write(OUT/'video_plan.json',plan)
    (OUT/'tests-before.log').write_text('PASS source trace hashes, selected physics/mission audit, 65-record summary, stop comparison 10-episode audit, initial pair equality, frame time ranges, 90 seconds and 2250 frames.\n',encoding='utf-8')
    with base.forbid():
        studio=Studio(media)
        try:
            # Exercise every motor highlight on saved geometry, without stepping physics.
            fake={**cases['rear'],'row':{**cases['rear']['row']}}
            for condition in JOINTS:
                fake['row']['condition']=condition
                studio.frame(fake,10000,(880,490))
                assert studio.highlights[-1]['highlight']==condition
            for chapter,seconds in CHAPTERS:
                frame=min(seconds*FPS-1,seconds*FPS//2)
                screen(media,studio,rows,cases,chapter,frame).save(OUT/f'preview_{chapter}.png')
        finally:studio.close()
    frozen(plan)
    print('Prepared 90-second English silent film; seven previews saved.',flush=True)


def render():
    plan=base.s.read(OUT/'video_plan.json');frozen(plan)
    target=OUT/'go2_rebalance_portfolio_90s.mp4';assert not target.exists()
    rows,cases=load();media=base.media_old.media_helper();indices=[];count=0
    writer=media.imageio_ffmpeg.write_frames(str(target),SIZE,fps=FPS,codec='libx264',quality=7,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-an','-movflags','+faststart','-threads','2'])
    writer.send(None)
    with base.forbid():
        studio=Studio(media)
        try:
            for chapter,seconds in CHAPTERS:
                static=screen(media,studio,rows,cases,chapter,0) if chapter=='results' else None
                for frame in range(seconds*FPS):
                    im=static if static is not None else screen(media,studio,rows,cases,chapter,frame)
                    writer.send(np.asarray(im));indices.append(dict(frame=count,chapter=chapter,ticks=frame_ticks(chapter,frame,cases)));count+=1
                    if frame in (0,seconds*FPS-1):im.save(OUT/f'{chapter}_{frame}.png')
                    if frame%125==0:print(f'{chapter} {frame}/{seconds*FPS}',flush=True)
        finally:writer.close();studio.close()
    assert count==2250
    write(OUT/'frame_indices.json',indices)
    write(OUT/'render_audit.json',dict(saved_pose_draws=studio.drawn,qpos_qvel_exact=True,
        physics_arrays_unchanged=True,actual_joint_highlights=studio.highlights,integration_steps=0,policy_predictions=0,training=False))
    frozen(plan)
    verify()


def verify():
    media=base.media_old.media_helper();exe=media.imageio_ffmpeg.get_ffmpeg_exe();path=OUT/'go2_rebalance_portfolio_90s.mp4'
    count,seconds=media.imageio_ffmpeg.count_frames_and_secs(str(path));assert count==2250 and abs(seconds-90)<.01
    reader=media.imageio_ffmpeg.read_frames(str(path));meta=next(reader);reader.close()
    assert tuple(meta['size'])==SIZE and meta['fps']==25
    proc=subprocess.run([exe,'-hide_banner','-xerror','-i',str(path),'-f','null','-'],capture_output=True,text=True,check=True)
    assert 'Audio:' not in proc.stderr
    times=[0,9.96,10,16,20,28,39.96,40,54.96,55,63,69.96,70,76,79.96,80,89.96]
    sheet=Image.new('RGB',(1920,360*((len(times)+2)//3)),'white')
    for i,t in enumerate(times):
        p=OUT/f'decoded_{i:02}.png'
        subprocess.run([exe,'-v','error','-y','-ss',str(t),'-i',str(path),'-frames:v','1',str(p)],capture_output=True,check=True)
        with Image.open(p) as im:
            assert np.std(np.asarray(im))>15
            sheet.paste(im.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(OUT/'contact_sheet.png')
    plan=base.s.read(OUT/'video_plan.json');frozen(plan)
    write(OUT/'verification.json',dict(passed=True,frames=count,seconds=seconds,size=SIZE,fps=25,silent=True,
        all_frames_decoded=True,source_files_preserved=len(plan['inputs_sha256']),video_sha256=base.s.digest(path),
        integration_steps=0,new_rollouts=0,policy_predictions=0,training=False))
    print(path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','render','verify'])
    {'prepare':prepare,'render':render,'verify':verify}[parser.parse_args().command]()
