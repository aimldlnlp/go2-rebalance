"""Data-driven film from saved states only; no integration or policy calls."""
from project_paths import project_file, source_files
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
import mujoco
from PIL import Image,ImageDraw
import make_rr_severity_visual_video as previous
import make_rr_severity_video as media_old
import sudden as s

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/walking_progress_data_video'
PREVIEW=ROOT/'work/walking_progress_data_video_preview'
FPS=25
SIZE=(1920,1080)
CHAPTERS=[('healthy',5),('comparison',20),('detail',10),('results',10)]
FOLDERS={'original':ROOT/'results/walking_residual_privileged_pilot',
         'progress':ROOT/'results/walking_progress_reward_pilot'}
LABELS={'original':'Original reward','progress':'More reward for moving forward'}
COLORS={'original':'#536c43','progress':'#a35317','healthy':'#536c43'}
INK='#222521';MUTED='#63665f';GRID='#e3e5e1';FAINT='#c9cec5'
SHORT={'FL_calf':'Front left','FR_calf':'Front right','RL_calf':'Rear left','RR_calf':'Rear right'}


def text(media,im,value,x,y,size=25,color=INK):
    media.put(im,value,x,y,size,fill=color)


def forbid():
    stack=previous.forbidden()
    from stable_baselines3 import PPO
    from walking_baseline import Actor
    stack.enter_context(patch.object(PPO,'predict',side_effect=AssertionError('No inference')))
    stack.enter_context(patch.object(PPO,'learn',side_effect=AssertionError('No training')))
    stack.enter_context(patch.object(Actor,'predict',side_effect=AssertionError('No nominal inference')))
    return stack


def derive(row,a):
    q,v=a['qpos'],a['qvel']
    assert np.isfinite(q).all() and np.isfinite(v).all()
    assert s.digest_bytes(q[0].tobytes()+v[0].tobytes())==row['initial_state_sha256']
    direction=np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw'])])
    side=np.array([-direction[1],direction[0]])
    delta=q[:,:2]-row['initial_xy']
    progress=delta@direction;lateral=delta@side
    error=np.linalg.norm(delta-2*direction,axis=1)
    qw,qx,qy,qz=q[:,3:7].T
    yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    heading=np.abs(np.arctan2(np.sin(yaw-row['initial_yaw']),np.cos(yaw-row['initial_yaw'])))
    planar=np.linalg.norm(v[:,:2],axis=1)
    stable=(error<=.15)&(heading<=np.pi/12)&(planar<=.05)
    hold=np.zeros(len(q));begin=None
    for i,ok in enumerate(stable):
        if not ok:begin=None
        elif begin is None:begin=i
        hold[i]=0. if begin is None else (i-begin)*.002
    assert abs(hold[-1]-row['final_stable_hold_s'])<1e-9
    np.testing.assert_allclose(error[-1],row['final_goal_error_m'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(progress[-1],row['final_progress_m'],rtol=0,atol=1e-12)
    return dict(row=row,qpos=q,qvel=v,progress=progress,lateral=lateral,error=error,hold=hold,
        speed=v[:,:2]@direction,planar_speed=planar,command=a['command'],
        first_goal_entry_s=next((i*.002 for i,x in enumerate(error) if x<=.15),None))


def bin_span(low,high,columns=100):
    assert low.shape==high.shape and low.shape[1]==12
    n=len(low);span=high-low
    assert np.isfinite(span).all() and np.all(span>=-1e-7) and np.all(span<=5.+1e-7)
    binned=np.full((12,columns),np.nan)
    for j in range(columns):
        start,end=j*10000//columns,min((j+1)*10000//columns,n)
        if start<end:binned[:,j]=span[start:end].mean(axis=0)
    return binned


def load():
    assert s.read(ROOT/'results/walking_torque_authority_review/final_verification.json')['passed']
    cases={};cohorts={};pins={};envelopes={}
    def pin(path):pins[str(path.relative_to(ROOT))]=s.digest(path)
    for name,folder in FOLDERS.items():
        proof=folder/'final_verification.json';pin(proof)
        assert s.read(proof)['passed'] if name=='progress' else s.read(proof)['audit_passed']
        path=folder/'validation_131072/episodes.json';pin(path)
        rows=s.read(path);assert len(rows)==25
        assert {(r['condition'],r['seed']) for r in rows}=={(c,seed) for c in ['healthy',*SHORT] for seed in range(23000,23005)}
        cohorts[name]=rows;cases[name]={}
        for row in rows:
            stem=f"{row['condition']}_{row['seed']}"
            path=folder/'validation_131072'/(stem+'.npz');pin(path);pin(path.with_suffix('.json'))
            assert pins[str(path.relative_to(ROOT))]==row['trace_sha256']
            assert row==s.read(path.with_suffix('.json')) and not row['interrupted']
            if row['condition']=='RR_calf':
                with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
                c=derive(row,a);assert c['qpos'].shape==(10001,19)
                c['row']={**row,'strength_remaining':.1,'historical_control':False}
                cases[name][row['seed']]=c
        pilot='progress10' if name=='original' else 'progress100'
        path=ROOT/f'results/walking_torque_authority_review/arrays/{pilot}_RR_calf_23000.npz';pin(path)
        metadata=s.read(ROOT/'results/walking_torque_authority_review/episodes.json')
        r=next(x for x in metadata if (x['pilot'],x['condition'],x['seed'])==(pilot,'RR_calf',23000))
        assert r['envelope_sha256']==s.digest(path) and r['source_trace_sha256']==cases[name][23000]['row']['trace_sha256']
        with np.load(path,allow_pickle=False) as z:envelopes[name]={k:z[k] for k in z.files}
        source=FOLDERS[name]/'validation_131072/RR_calf_23000.npz'
        with np.load(source,allow_pickle=False) as a:
            np.testing.assert_array_equal(envelopes[name]['actual'],a['applied_torque'])
        envelopes[name]['binned']=bin_span(envelopes[name]['low'],envelopes[name]['high'])
    path=ROOT/'results/walking_rr_severity/historical_controls.json';pin(path)
    row=next(r for r in s.read(path) if r['condition']=='healthy' and r['seed']==23000)
    path=ROOT/row['trace'];pin(path);assert s.digest(path)==row['trace_sha256']
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    healthy=derive(row,a)
    for name in cases:
        np.testing.assert_array_equal(cases[name][23000]['qpos'][0],healthy['qpos'][0])
        np.testing.assert_array_equal(cases[name][23000]['qvel'][0],healthy['qvel'][0])
    totals={name:dict(success=sum(r['success'] for r in rows if r['condition']!='healthy'),
        falls=sum(r['fallen'] for r in rows if r['condition']!='healthy')) for name,rows in cohorts.items()}
    assert totals=={'original':{'success':0,'falls':7},'progress':{'success':0,'falls':6}}
    for name in cases:
        assert all(not c['row']['fallen'] and not c['row']['success'] for c in cases[name].values())
    return dict(cases=cases,cohorts=cohorts,healthy=healthy,envelopes=envelopes,totals=totals,pins=pins)


def ticks(chapter,frame,data):
    if chapter=='healthy':return {'healthy':min(frame*80,10000)}
    if chapter=='comparison':return {name:min(frame*20,10000) for name in FOLDERS}
    if chapter=='detail':
        return {name:min((data['cases'][name][23000]['row']['fault_onset_tick']//10+1)*10+frame*10,10000) for name in FOLDERS}
    return {name:10000 for name in FOLDERS}


class Studio(previous.VisualStudio):
    def frame(self,case,tick,size,color):
        b=self.base;m=b.model
        if self.size!=size:
            b.renderer.close();b.renderer=mujoco.Renderer(m,width=size[0],height=size[1]);self.size=size
        wide=case['row']['condition']=='healthy'
        m.vis.global_.fovy=48
        b.camera.lookat[:]=[1. if wide else .48,0.,.16]
        b.camera.distance=2.65 if wide else 1.42
        b.camera.azimuth=105;b.camera.elevation=-30
        row=case['row'];onset=row['fault_onset_tick']
        active=onset is not None and tick>=onset
        media_old.previous.highlight(b,b.original,'RR_calf' if active else 'healthy')
        b.data.qpos[:]=case['qpos'][tick];b.data.qvel[:]=case['qvel'][tick]
        mujoco.mj_forward(m,b.data)
        np.testing.assert_array_equal(b.data.qpos,case['qpos'][tick])
        np.testing.assert_array_equal(b.data.qvel,case['qvel'][tick])
        b.renderer.update_scene(b.data,camera=b.camera)
        b.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
        b.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        start=np.r_[row['initial_xy'],.009]
        f=np.array([np.cos(row['initial_yaw']),np.sin(row['initial_yaw']),0.])
        side=np.array([-f[1],f[0],0.]);goal=start+2*f
        for value in np.arange(0,2,.15):
            for cross in [-.2,.2]:
                self.line(start+value*f+cross*side,start+min(value+.06,2)*f+cross*side,[.6,.7,.75,1],.002)
        self.line(goal-.32*side,goal+.32*side,[.63,.82,.58,1],.007)
        for angle in np.linspace(0,2*np.pi,49)[:-1]:
            self.line(goal+.15*(np.cos(angle)*f+np.sin(angle)*side),
                goal+.15*(np.cos(angle+np.pi/24)*f+np.sin(angle+np.pi/24)*side),[.63,.82,.58,1],.004)
        rgb=tuple(int(color[i:i+2],16)/255 for i in [1,3,5])+(1.,)
        points=case['qpos'][np.unique(np.r_[np.arange(0,tick+1,80),tick]),:2]
        for a,z in zip(points[:-1],points[1:]):
            if np.linalg.norm(z-a)>1e-7:self.line(np.r_[a,.015],np.r_[z,.015],rgb,.005)
        if active:
            knee=b.data.xpos[m.body('RR_calf').id].copy()
            self.line(knee,knee+[0,0,.08],[.94,.61,.2,1],.008)
        image=Image.fromarray(b.renderer.render());self.drawn+=1
        return image


def space_time(media,im,cases,now,box):
    x,y,w,h=box;draw=ImageDraw.Draw(im)
    origin=np.array([x+w*.27,y+h*.80])
    ux=np.array([w*.57,h*.08]);uy=np.array([-w*.20,h*.12]);ut=np.array([0.,-h*.64])
    def project(forward,side,time):return origin+ux*(forward/2)+uy*(side/.2)+ut*(time/20)
    for t in [0,5,10,15,20]:
        coords=[project(a,b,t) for a,b in [(0,-.2),(2,-.2),(2,.2),(0,.2),(0,-.2)]]
        draw.line([tuple(p) for p in coords],fill=GRID,width=1)
        label=project(0,.2,t)+[-25,-6]
        text(media,im,str(t),*label,17,MUTED)
    goal_plane=[project(2,b,t) for b,t in [(-.2,0),(.2,0),(.2,20),(-.2,20),(-.2,0)]]
    draw.line([tuple(p) for p in goal_plane],fill='#bdcbb3',width=2)
    text(media,im,'Goal 2 m',x+w*.71,y+h*.78,18,MUTED)
    for a,b,label in [(project(0,0,0),project(2,0,0),'Forward (m)'),
                      (project(0,0,0),project(0,.2,0),'Side (m)'),
                      (project(0,0,0),project(0,0,20),'Time (s)')]:
        draw.line((tuple(a),tuple(b)),fill='#aeb7ad',width=1)
    text(media,im,'Forward (m)',x+w*.55,y+h*.91,18,MUTED)
    text(media,im,'Side (m)',x,y+h*.85,18,MUTED)
    text(media,im,'Time (s)',x+w*.13,y+5,18,MUTED)
    text(media,im,'2',*project(2,0,0),18,MUTED)
    for name,case in cases.items():
        tick=now[name];color=COLORS[name]
        indices=np.unique(np.r_[np.arange(0,tick+1,40),tick])
        xy=[tuple(project(case['progress'][i],case['lateral'][i],i*.002)) for i in indices]
        if len(xy)>1:draw.line(xy,fill=color,width=3)
        px,py=xy[-1];draw.ellipse((px-5,py-5,px+5,py+5),fill=color,outline='white',width=1)
    text(media,im,'±0.20 m',x,y+h*.93,16,MUTED)


def curve(media,im,data,now,field,box,title):
    x,y,w,h=box;draw=ImageDraw.Draw(im)
    text(media,im,title,x,y,25)
    left,top,right,bottom=x+48,y+36,x+w-12,y+h-26
    if field=='error':low,high=0.,2.1
    else:
        allvalues=np.concatenate([c['speed'] for group in data['cases'].values() for c in group.values()])
        low=min(-.05,float(allvalues.min())-.02);high=max(.25,float(allvalues.max())+.02)
    def xy(t,value):return left+(right-left)*t/20,bottom-(bottom-top)*(value-low)/(high-low)
    levels=[0,1,2] if field=='error' else np.arange(np.ceil(low*10),np.floor(high*10)+1)/10
    for value in levels:
        a,b=xy(0,value),xy(20,value)
        draw.line((a,b),fill=GRID,width=1)
        text(media,im,f'{value:g}',x,y+h-29-(bottom-top)*(value-low)/(high-low),17,MUTED)
    draw.line((left,top,left,bottom,right,bottom),fill='#9ca596',width=1)
    for name,group in data['cases'].items():
        tick=now[name]
        for seed,case in group.items():
            inds=np.arange(0,10001,40)
            values=case[field] if field=='error' else case['speed']
            points=[xy(i*.002,values[i]) for i in inds]
            pale='#b7c3ae' if name=='original' else '#d7b798'
            draw.line(points,fill=pale,width=1)
            if seed==23000:
                inds=np.unique(np.r_[np.arange(0,tick+1,20),tick])
                draw.line([xy(i*.002,values[i]) for i in inds],fill=COLORS[name],width=3)
                px,py=xy(tick*.002,values[tick]);draw.ellipse((px-4,py-4,px+4,py+4),fill=COLORS[name])
        at=xy(tick*.002,0)[0]
        draw.line((at,top,at,bottom),fill=COLORS[name],width=1)
    if field=='error':
        py=xy(0,.15)[1]
        for at in np.arange(left,right,12):draw.line((at,py,min(at+5,right),py),fill=MUTED)
        text(media,im,'Goal area',right-94,py-22,17,MUTED)
    text(media,im,'0',left,bottom+6,17,MUTED);text(media,im,'20 s',right-34,bottom+6,17,MUTED)


def heatmap(media,im,binned,tick,box,title,size=21):
    x,y,w,h=box;draw=ImageDraw.Draw(im)
    text(media,im,title,x,y,size)
    gx,gy=x+128,y+43;gw=w-142;rh=(h-95)/12
    rgb=np.array([83,108,67]);background=np.array([239,241,237])
    for row in range(12):
        for col in range(100):
            value=binned[row,col]
            color='#e5e5e5' if not np.isfinite(value) else tuple(np.rint(background+(rgb-background)*np.clip(value/5,0,1)).astype(int))
            draw.rectangle((gx+gw*col/100,gy+rh*row,gx+gw*(col+1)/100+1,gy+rh*(row+1)-2),fill=color)
    for j,label in enumerate(['Front left','Front right','Rear left','Rear right']):
        text(media,im,label,x,gy+(j*3+.7)*rh,18,MUTED)
        if j:draw.line((gx,gy+j*3*rh-2,gx+gw,gy+j*3*rh-2),fill='white',width=2)
    cursor=gx+gw*tick/10000
    draw.line((cursor,gy-4,cursor,gy+12*rh),fill='#a35317',width=2)
    text(media,im,'0',gx,gy+12*rh+4,17,MUTED);text(media,im,'20 s',gx+gw-35,gy+12*rh+4,17,MUTED)
    text(media,im,'Hip / upper leg / knee',x,y+h-27,18,MUTED)


def outcomes(media,im,data,box,large=False):
    x,y,w,h=box;draw=ImageDraw.Draw(im)
    text(media,im,'All four weakened knees',x,y,31 if large else 25)
    text(media,im,'Five starting poses each',x,y+40,22 if large else 19,MUTED)
    left,right=x+w*.49,x+w*.85
    text(media,im,'Original',left-36,y+83,23 if large else 20,COLORS['original'])
    text(media,im,'More progress',right-70,y+83,23 if large else 20,COLORS['progress'])
    top=y+125;rh=(h-245)/20
    for group,condition in enumerate(SHORT):
        text(media,im,SHORT[condition],x,top+(group*5+1.5)*rh,22 if large else 19,MUTED)
        for j,seed in enumerate(range(23000,23005)):
            py=top+(group*5+j)*rh
            draw.line((left,py,right,py),fill=GRID,width=1)
            for name,px in [('original',left),('progress',right)]:
                row=next(r for r in data['cohorts'][name] if (r['condition'],r['seed'])==(condition,seed))
                color='#a35317' if row['fallen'] else '#c1c6bd'
                radius=5 if large else 4
                draw.ellipse((px-radius,py-radius,px+radius,py+radius),fill=color)
        if group:draw.line((x,top+group*5*rh-rh*.5,x+w,top+group*5*rh-rh*.5),fill=GRID,width=1)
    text(media,im,'Completed',x,y+h-100,25 if large else 22)
    text(media,im,'0/20',left-26,y+h-100,30 if large else 25)
    text(media,im,'0/20',right-26,y+h-100,30 if large else 25)
    text(media,im,'Falls',x,y+h-52,25 if large else 22)
    text(media,im,'7/20',left-26,y+h-52,30 if large else 25,COLORS['progress'])
    text(media,im,'6/20',right-26,y+h-52,30 if large else 25,COLORS['progress'])


def hold_ring(media,im,value,x,y):
    draw=ImageDraw.Draw(im);box=(x,y,x+100,y+100)
    draw.arc(box,-90,270,fill=GRID,width=8)
    if value>0:draw.arc(box,-90,-90+360*min(value/2,1),fill=COLORS['healthy'],width=8)
    text(media,im,'2 s',x+30,y+34,27)


def screen(media,studio,data,chapter,frame):
    im=Image.new('RGB',SIZE,'white');draw=ImageDraw.Draw(im);now=ticks(chapter,frame,data)
    text(media,im,'Walking with a weakened knee',40,20,42)
    if chapter=='healthy':
        c=data['healthy'];tick=now['healthy']
        text(media,im,'Healthy reference · Archive',40,88,30,COLORS['healthy'])
        im.paste(studio.frame(c,tick,(1240,690),COLORS['healthy']),(40,140))
        text(media,im,'Movement over time',1320,160,30)
        space_time(media,im,{'healthy':c},{'healthy':tick},(1310,215,560,480))
        hold_ring(media,im,c['hold'][tick],1390,730)
        text(media,im,'Hold for 2 seconds',1520,766,27)
        text(media,im,'2 m in 20 s     Goal radius 0.15 m     Heading ±15°     Path ±0.20 m',40,880,28)
        text(media,im,'Hold position and heading for 2 s · speed at most 0.05 m/s · no fall',40,930,27,MUTED)
        text(media,im,f'Time {tick*.002:.2f} s   ·   4× playback',40,1018,25,MUTED)
        return im
    if chapter=='comparison':
        main={name:data['cases'][name][23000] for name in FOLDERS}
        for name,x in [('original',40),('progress',1290)]:
            c=main[name];tick=now[name]
            text(media,im,LABELS[name],x,88,27,COLORS[name])
            im.paste(studio.frame(c,tick,(590,360),COLORS[name]),(x,135))
            strength=10 if tick>=c['row']['fault_onset_tick'] else 100
            text(media,im,f'Rear-right knee · {strength}% strength',x,505,23,MUTED)
        text(media,im,'Movement over time',686,90,29)
        space_time(media,im,main,now,(650,145,600,350))
        curve(media,im,data,now,'error',(40,565,590,200),'Distance to goal (m)')
        curve(media,im,data,now,'speed',(40,800,590,200),'Forward speed (m/s)')
        heatmap(media,im,data['envelopes']['original']['binned'],now['original'],(665,565,570,212),'Room to adjust · Original',23)
        heatmap(media,im,data['envelopes']['progress']['binned'],now['progress'],(665,795,570,212),'Room to adjust · More progress',23)
        outcomes(media,im,data,(1290,565,580,440))
        text(media,im,f'Time {now["original"]*.002:05.2f} / 20 s   ·   1× playback',40,1030,22,MUTED)
        text(media,im,'Turning-force range: pale 0 → green 5 Nm',665,1030,19,MUTED)
        text(media,im,'Amber: fell · Gray: stayed upright',1290,1030,19,MUTED)
        return im
    if chapter=='detail':
        text(media,im,'After the knee weakens',40,86,30)
        for name,x in [('original',40),('progress',1000)]:
            c=data['cases'][name][23000];tick=now[name]
            text(media,im,LABELS[name],x,138,29,COLORS[name])
            im.paste(studio.frame(c,tick,(880,380),COLORS[name]),(x,185))
            heatmap(media,im,data['envelopes'][name]['binned'],tick,(x,590,880,310),'Room to adjust',28)
            torque=min(tick,9999);e=data['envelopes'][name]
            span=e['high'][torque,11]-e['low'][torque,11]
            text(media,im,f'Weakened knee · turning-force range {span:.3f} Nm',x,932,26,COLORS[name])
        text(media,im,f'Time since weakening {frame*.02:04.2f} s   ·   0.5× playback',40,1026,23,MUTED)
        text(media,im,'At each recorded moment · pale 0 → green 5 Nm',1000,1026,22,MUTED)
        return im
    text(media,im,'Saved outcomes · All starting poses',40,88,31)
    for name,y in [('original',155),('progress',565)]:
        text(media,im,LABELS[name],40,y,28,COLORS[name])
        im.paste(studio.frame(data['cases'][name][23000],10000,(690,335),COLORS[name]),(40,y+45))
    outcomes(media,im,data,(830,150,1020,800),True)
    text(media,im,'Amber: fell · Gray: stayed upright · No completed fault missions',830,978,23,MUTED)
    text(media,im,'Saved flat-floor simulation · One training run per method · Other conditions not tested',40,1030,23,MUTED)
    return im


def preview():
    assert not PREVIEW.exists()
    PREVIEW.mkdir();data=load();media=media_old.media_helper()
    with forbid():
        studio=Studio(media)
        try:
            for chapter,frame in [('healthy',120),('comparison',375),('detail',150),('results',0)]:
                screen(media,studio,data,chapter,frame).save(PREVIEW/f'{chapter}.png')
        finally:studio.close()
    print(PREVIEW,flush=True)


def prepare():
    assert not OUT.exists()
    assert 'OK' in (ROOT/'work/progress_data_video_tests.log').read_text(encoding='utf-8')
    data=load();media=media_old.media_helper()
    paths=[p for p in (ROOT/'results').rglob('*') if p.is_file()]+list(source_files(ROOT))
    paths+=[media_old.previous.HELPER,ROOT/'work/progress_data_video_tests.log']
    paths+=[p for p in (ROOT/'vendor/mujoco_menagerie/unitree_go2').rglob('*') if p.is_file()]
    font=Path(r'C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf')
    paths.append(font)
    protected={};external={}
    for i,path in enumerate(paths):
        path=Path(path)
        if path.is_relative_to(ROOT):protected[str(path.relative_to(ROOT))]=s.digest(path)
        else:external[str(path)]=s.digest(path)
        if (i+1)%500==0:print(f'Protected inputs: {i+1}',flush=True)
    OUT.mkdir()
    s.write(OUT/'video_plan.json',dict(size=SIZE,fps=FPS,chapters=CHAPTERS,seconds=45,frames=1125,
        source_inputs=data['pins'],protected_inputs=protected,external_helpers=external,
        motion='Exact saved qpos/qvel, geometry-only mj_forward, all integrators and policy calls forbidden',
        data='Space-time path, floor ribbons, per-pose distance and forward velocity, bin-mean torque span heatmaps and actual paired outcomes',
        scope='Healthy archived nominal reference; pose23000 rear-right knee in both trained arms; all20 fault poses in summary',
        curve_strands='Five actual saved poses each arm; highlighted pose23000; future strands are already-saved data, not forecasts',
        render_runtime=dict(mujoco=mujoco.__version__,numpy=np.__version__,encoder=media.imageio_ffmpeg.get_ffmpeg_exe()),
        heatmap='12 joints ordered hip/upper leg/knee in four leg groups; 100 equal0.2s bins mean instantaneous per-joint high-low Nm; scale0to5 common',
        no_inferred_contact_forces=True,training=False,new_rollouts=0,new_policy_inference=0))
    s.write(OUT/'source_data_verification.json',dict(passed=True,all_initial_poses_exact=True,
        totals=data['totals'],main_trace_states=110011,main_motion_all_full20s=True,
        endpoint_metrics_recomputed=True,torque_envelope_source_hashes_checked=True))
    (OUT/'storyboard.md').write_text('''# Saved-data video storyboard

- 0–5s: archived healthy nominal walker at4x. Goal ring, measured hold,3D saved path and short mission criteria.
- 5–25s: the two reward variants at1x, same starting pose23000, each own recorded fault onset. World-space paths, synchronized3D forward/side/time plot, five saved curves per arm, two torque-span heatmaps and all20 paired outcomes. Numeric clock is simulation time.
- 25–35s: first5s after first informed control, each arm aligned to own actual latch boundary, at0.5x. Larger heatmaps show the instantaneous saved-state torque interval. The displayed span is a calculation at that state, not measured compensation benefit.
- 35–45s: exact saved end poses and all20 fault pairs. Amber is a fall; gray is upright, not mission completion. The summary explicitly states0/20 completion in both arms and7/20 versus6/20 falls.

White/LatinModernRoman, blue checker MuJoCo arenas, muted green/orange identity, plain English captions. No synthesized contacts, predictions, new physics, policy inference or training. Curve and heatmap data use original500Hz states; display frames25Hz. No scientific claims from ImageGen assets; preview artwork is not used in the final film.
''',encoding='utf-8')
    print('Frozen45s/1080p silent film',flush=True)


def verify_inputs(plan):
    for name,sha in plan['protected_inputs'].items():assert s.digest(project_file(ROOT, name))==sha,name
    for name,sha in plan['external_helpers'].items():assert s.digest(name)==sha,name


def render():
    p=s.read(OUT/'video_plan.json');verify_inputs(p)
    assert not (OUT/'walking_progress_data.mp4').exists()
    media=media_old.media_helper();data=load();indices=[];count=0
    target=OUT/'walking_progress_data.mp4'
    writer=media.imageio_ffmpeg.write_frames(str(target),SIZE,fps=FPS,codec='libx264',quality=7,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-an','-movflags','+faststart','-threads','2'])
    writer.send(None)
    with forbid():
        studio=Studio(media)
        try:
            for chapter,seconds in CHAPTERS:
                static=screen(media,studio,data,chapter,0) if chapter=='results' else None
                for frame in range(seconds*FPS):
                    im=static if static is not None else screen(media,studio,data,chapter,frame)
                    writer.send(np.asarray(im));indices.append(dict(frame=count,chapter=chapter,ticks=ticks(chapter,frame,data)))
                    count+=1
                    if frame in [0,seconds*FPS-1]:im.save(OUT/f'{chapter}_{frame}.png')
                    if frame%125==0:print(f'{chapter} {frame}/{seconds*FPS}',flush=True)
        finally:writer.close();studio.close()
    assert count==1125
    s.write(OUT/'frame_indices.json',indices)
    s.write(OUT/'render_verification.json',dict(frames=count,rendered_saved_poses=studio.drawn,
        qpos_qvel_exact_every_draw=True,physics_arrays_unchanged=True,integration_steps=0,
        policy_predictions=0,training=False,model_version=mujoco.__version__))
    verify()


def verify():
    media=media_old.media_helper();exe=media.imageio_ffmpeg.get_ffmpeg_exe();path=OUT/'walking_progress_data.mp4'
    count,duration=media.imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count==1125 and abs(duration-45)<.01
    reader=media.imageio_ffmpeg.read_frames(str(path));meta=next(reader);reader.close()
    assert tuple(meta['size'])==SIZE and meta['fps']==FPS
    proc=subprocess.run([exe,'-hide_banner','-i',str(path),'-f','null','-'],capture_output=True,text=True,check=True)
    assert 'Audio:' not in proc.stderr
    subprocess.run([exe,'-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,check=True)
    sheet=Image.new('RGB',SIZE,'white')
    for i,t in enumerate([0,4.96,5,14,24.96,25,31,35,44.96]):
        output=OUT/f'decoded_{i}.png'
        subprocess.run([exe,'-v','error','-ss',str(t),'-i',str(path),'-frames:v','1',str(output)],check=True,capture_output=True)
        with Image.open(output) as im:
            assert np.std(np.asarray(im))>15
            sheet.paste(im.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(OUT/'contact_sheet.png')
    p=s.read(OUT/'video_plan.json');verify_inputs(p)
    s.write(OUT/'verification.json',dict(passed=True,frames=count,seconds=duration,size=SIZE,fps=FPS,
        silent=True,all_frames_decoded=True,protected_inputs_unchanged=len(p['protected_inputs']),
        source_traces_hash_checked=True,exact_saved_motion=True,
        video_sha256=s.digest(path),new_rollouts=0,training=False,new_policy_inference=0))
    (OUT/'README.md').write_text('''# Walking comparison with linked data views

Open walking_progress_data.mp4:45s,1920×1080,25fps,silent. LatinModernRoman and original MuJoCo robot geometry/checker floor. Archived healthy reference at4x; main paired motion at1x; post-latch detail at0.5x; final paired outcomes.

Main motion shows pose23000, rear-right knee at10% strength after each actual fault latch. Original reward and more-progress-reward arms are the privileged72-input pilots at131072 steps, weights10 and100 respectively; those internal names are omitted from on-screen captions. Initial states match, but trained policies can differ before fault. Motion is direct saved qpos/qvel, with zero integration and zero policy inference.

The3D plot uses actual forward distance, side displacement and simulation time. Different axis units/scales are explicitly labelled. There are no generated or predicted trajectories. Five faint lines per method are the five actual development poses; the bold line/markers show23000. Heatmaps average instantaneous torque-envelope width in100 bins of0.2s, for all12 joints. Rows within each leg are hip/upper leg/knee, common scale0–5Nm. A zero width means no instantaneous torque change is available inside±0.10rad at that saved state, not inability to walk globally. Per-state variation inside a bin is averaged. In detail the exact pointwise weak-knee span is also printed.

All20 fault pairs are grouped by the actual weakened knee, five starting poses each. Gray dots mean no fall, not success; amber means fall. Both arms have0/20 completed missions, and7/20 versus6/20 falls. The archived healthy reference is not one of these20 pairs. A complete hold indicator alone does not certify all mission checks.

Mission:2m within20s; position error≤0.15m, heading≤15°, cross-track≤0.20m; final continuous2s inside position/heading with planar speed≤0.05m/s; no fall or nonfinite. Fall thresholds height0.16m/tilt45°. Physics.002s, original control.02s, recorded fault after limited torque at progress0.50m. White lane guides refer to body cross-track, not foot placement.

Only final saved states/results are shown in the closing chapter, which is a labelled static results view. Detail aligns each method to its own first control observation after actual latch. Headings use plain words; no ImageGen geometry or illustrative curves are consumed. Original artifacts are retained and rehashed. verification.json documents full-stream decode and protected inputs; frame_indices.json maps every encoded frame to exact source indices. Visual QA samples are documented separately; no claim of full manual playback.

Limits: one training seed, reused development poses, flat-floor simulation; no generalization or hardware evidence. The film does not identify a single cause of failure or establish feasible recovery. All new work is rendering and arithmetic on saved traces.
''',encoding='utf-8')
    print(path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preview','prepare','render','verify'])
    args=parser.parse_args();{'preview':preview,'prepare':prepare,'render':render,'verify':verify}[args.command]()
