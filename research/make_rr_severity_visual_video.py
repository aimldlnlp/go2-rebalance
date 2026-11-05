"""Visual revision of saved RR motion. Rendering only; never integrates physics."""
from project_paths import project_file
import argparse
from contextlib import ExitStack
from unittest.mock import patch
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import make_rr_severity_video as old
import diagnose_walking_fault as d

ROOT=old.ROOT
OUT=ROOT/'results/walking_rr_severity_video_visual'
PREVIEW=ROOT/'work/rr_severity_visual_previews'
FPS=25
SIZE=(1920,1080)
CHAPTERS=[('paired',20),('levels',10),('hold',8),('results',4)]
BG='#ffffff'
INK='#222521'
MUTED='#63665f'
GREEN='#536c43'
AMBER='#a35317'


def frame_tick(chapter,frame):
    if chapter=='paired':return min(10000,frame*20)
    if chapter=='levels':return min(10000,frame*40)
    if chapter=='hold':return min(10000,8000+frame*10)
    return 10000


def indicators(case,tick):
    row=case['row'];onset=row['fault_onset_tick']
    active=onset is not None and tick>=onset
    return dict(time_s=tick*.002,progress_m=float(case['progress'][tick]),
                hold_s=float(case['hold'][tick]),hold_fraction=float(np.clip(case['hold'][tick]/2,0,1)),
                strength= row['strength_remaining'] if active else 1.,fault_active=active,
                in_goal=bool(case['error'][tick]<=.15))


def validate_indicators(cases):
    """Independent full-resolution checks, including position, heading and speed."""
    proof={}
    for name,c in cases.items():
        q,v=c['qpos'],c['qvel'];r=c['row']
        goal=np.asarray(r['initial_xy'])+2*np.array([np.cos(r['initial_yaw']),np.sin(r['initial_yaw'])])
        error=np.sqrt(np.sum((q[:,:2]-goal)**2,axis=1))
        np.testing.assert_allclose(error,c['error'],atol=1e-14,rtol=0)
        yaw=np.array([np.arctan2(2*(a*d_+b*c_),1-2*(c_*c_+d_*d_)) for a,b,c_,d_ in q[:,3:7]])
        delta=(yaw-r['initial_yaw']+np.pi)%(2*np.pi)-np.pi
        stable=(error<=.15)&(np.abs(delta)<=np.pi/12)&(np.sum(v[:,:2]**2,axis=1)<=.05**2)
        index=np.arange(len(q));reset=np.maximum.accumulate(np.where(~stable,index,-1))
        # First stable state starts a zero-duration interval, matching the mission evaluator.
        starts=reset+1
        hold=np.where(stable,(index-starts)*.002,0.)
        np.testing.assert_allclose(hold,c['hold'],rtol=0,atol=1e-12)
        assert abs(hold[-1]-r['final_stable_hold_s'])<1e-9
        onset=r['fault_onset_tick']
        if onset is not None:
            assert np.all(c['progress'][:onset]<.5) and c['progress'][onset]>=.5
            assert not indicators(c,onset-1)['fault_active'] and indicators(c,onset)['fault_active']
        for tick in range(len(q)):
            value=indicators(c,tick)
            assert value['hold_fraction']==min(hold[tick]/2,1.)
        proof[name]=dict(states=len(q),final_hold_s=float(hold[-1]),final_error_m=float(error[-1]),
                         first_goal_entry_s=c['first_goal_entry_s'],fault_onset_tick=onset,
                         trace_sha256=r['trace_sha256'])
    return proof


class VisualStudio:
    def __init__(self,media):
        self.base=old.studio_for(media)
        m=self.base.model;original=self.base.original
        self.physics={key:getattr(m,key).copy() for key in ['geom_friction','geom_size','geom_pos',
                     'body_mass','body_inertia','dof_damping','dof_frictionloss','jnt_range','actuator_gear']}
        self.timestep=m.opt.timestep
        m.tex_data[:]=original.tex_data
        floor=m.geom('floor').id
        m.geom_matid[floor]=original.geom_matid[floor]
        m.geom_rgba[floor]=original.geom_rgba[floor]
        m.mat('groundplane').reflectance[:]=0
        m.vis.headlight.ambient[:]=[.2]*3;m.vis.headlight.diffuse[:]=[.45]*3
        m.light_castshadow[:]=True
        m.light_dir[0]=[-.35,.3,-1]
        m.stat.center[:]=[1.,0.,.2]
        m.stat.extent=3.5
        m.vis.map.shadowscale=.8
        m.vis.map.shadowclip=1.5
        m.vis.quality.shadowsize=4096
        m.vis.rgba.haze[:]=original.vis.rgba.haze
        self.base.camera.azimuth=105
        self.base.camera.elevation=-33
        self.base.camera.distance=3.05
        self.size=None;self.drawn=0

    def line(self,a,b,color,width=.006):
        scene=self.base.renderer.scene
        geom=scene.geoms[scene.ngeom];scene.ngeom+=1
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color)
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,width,a,b)

    def frame(self,case,tick,size,detail=False):
        b=self.base;m=b.model
        if self.size!=size:
            b.renderer.close();b.renderer=mujoco.Renderer(m,width=size[0],height=size[1]);self.size=size
        r=case['row'];f=np.array([np.cos(r['initial_yaw']),np.sin(r['initial_yaw']),0.])
        side=np.array([-f[1],f[0],0.]);start=np.r_[r['initial_xy'],.007];goal=start+2*f
        b.camera.lookat[:]=start+f*(1.85 if detail else 1.)+[0,0,.12]
        b.camera.distance=1.45 if detail else (1.7 if size[1]==330 else 3.5 if size[0]==360 else 2.7)
        old.previous.highlight(b,b.original,'RR_calf' if indicators(case,tick)['fault_active'] else 'healthy')
        b.data.qpos[:]=case['qpos'][tick];b.data.qvel[:]=case['qvel'][tick]
        mujoco.mj_forward(m,b.data)
        np.testing.assert_array_equal(b.data.qpos,case['qpos'][tick])
        np.testing.assert_array_equal(b.data.qvel,case['qvel'][tick])
        b.renderer.update_scene(b.data,camera=b.camera)
        b.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
        b.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        for t in np.arange(0,2,.12):self.line(start+t*f,start+min(t+.05,2)*f,[.7,.8,.85,1],.003)
        self.line(goal-.34*side,goal+.34*side,[.55,.9,.7,1],.009)
        for theta in np.linspace(0,2*np.pi,41)[:-1]:
            a=goal+.15*(np.cos(theta)*f+np.sin(theta)*side)
            z=theta+2*np.pi/40
            end=goal+.15*(np.cos(z)*f+np.sin(z)*side)
            self.line(a,end,[.55,.9,.7,1],.004)
        knee=b.data.xpos[m.body('RR_calf').id].copy()
        color=[.94,.61,.2,1] if indicators(case,tick)['fault_active'] else [.7,.8,.85,1]
        self.line(knee,knee+[0,0,.07],color,.009)
        image=Image.fromarray(b.renderer.render());self.drawn+=1
        return image

    def close(self):
        for key,value in self.physics.items():np.testing.assert_array_equal(getattr(self.base.model,key),value)
        assert self.base.model.opt.timestep==self.timestep
        self.base.close()


def text(media,image,value,x,y,size=26,color=INK):
    media.put(image,value,x,y,size,fill=color)


def gauge(media,image,value,x,y,width=260):
    draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((x,y,x+width,y+9),radius=4,fill='#e1e5dd')
    if value>0:draw.rounded_rectangle((x,y,x+width*min(value/2,1),y+9),radius=4,fill=GREEN)
    text(media,image,f'{min(value,2):.2f} / 2 s',x,y+17,24,GREEN if value>=2 else MUTED)


def timeline(media,image,tick):
    draw=ImageDraw.Draw(image);x,y,w=55,1010,1810
    draw.rectangle((x+w*.9,y-9,x+w,y+9),fill='#dde8d6')
    draw.line((x,y,x+w,y),fill='#6d8493',width=2)
    at=x+w*tick/10000
    draw.ellipse((at-7,y-7,at+7,y+7),fill=INK)
    text(media,image,'0',x,1030,23,MUTED);text(media,image,'18',x+w*.9-12,1030,23,MUTED)
    text(media,image,'20 s',x+w-46,1030,23,MUTED)


def screen(media,studio,cases,chapter,tick):
    im=Image.new('RGB',SIZE,BG)
    title={'paired':'Healthy / RR 75%','levels':'RR strength','hold':'Goal / hold','results':'Mission success'}[chapter]
    text(media,im,title,50,24,39)
    speed={'paired':'1x','levels':'2x','hold':'0.5x','results':'5 poses'}[chapter]
    text(media,im,speed,1690,34,28,MUTED)
    if chapter=='results':
        for i,name in enumerate(old.ORDER):
            x=38+i*378;c=cases[name]
            im.paste(studio.frame(c,10000,(360,400)),(x,200))
            text(media,im,old.LABELS[name],x+15,140,30,GREEN if name=='healthy' else AMBER)
            count=c['summary']['successes'];draw=ImageDraw.Draw(im)
            for j in range(5):
                px=x+42+j*57;draw.ellipse((px,670,px+24,694),fill=GREEN if j<count else '#ffffff',outline=GREEN if j<count else MUTED,width=2)
            text(media,im,f'{count}/5',x+140,730,44)
        text(media,im,'Healthy + RR 10%: archive',50,950,24,MUTED)
        return im
    names=['healthy','rr_75'] if chapter!='levels' else old.ORDER[1:]
    for i,name in enumerate(names):
        c=cases[name];value=indicators(c,tick)
        x=35+(i%2)*960
        if chapter=='levels':
            y=100+(i//2)*425;size=(900,330)
        else:y=120;size=(900,650)
        im.paste(studio.frame(c,tick,size,detail=chapter=='hold'),(x,y+45))
        label='Healthy' if name=='healthy' else f"RR {round(value['strength']*100)}%"
        text(media,im,label,x+12,y,30,GREEN if name=='healthy' else AMBER)
        if c['row']['historical_control']:text(media,im,'archive',x+740,y+4,22,MUTED)
        bottom=y+45+size[1]
        if chapter=='levels':
            text(media,im,f"{value['progress_m']:.2f} m / 2 m",x+15,bottom+4,25)
            gauge(media,im,value['hold_s'],x+550,bottom+5,300)
        else:
            text(media,im,f"{value['progress_m']:.2f} m / 2 m",x+15,bottom+26,32)
            text(media,im,'Hold',x+550,bottom+16,23,MUTED)
            gauge(media,im,value['hold_s'],x+550,bottom+53,300)
    if chapter=='hold':text(media,im,'Goal radius 0.15 m',50,953,25,MUTED)
    else:text(media,im,'Goal 2 m   /   radius 0.15 m',50,953,25,MUTED)
    text(media,im,f't = {tick*.002:05.2f} s',1580,953,29)
    timeline(media,im,tick)
    return im


def forbidden():
    stack=ExitStack()
    for name in ['mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta']:
        stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError('Rendering only: integration forbidden')))
    return stack


def preview():
    assert not PREVIEW.exists()
    PREVIEW.mkdir();media=old.media_helper();cases=old.load_cases()
    with forbidden():
        proof=validate_indicators(cases);studio=VisualStudio(media)
        try:
            for chapter,tick in [('paired',9500),('levels',8500),('hold',10000),('results',10000)]:
                screen(media,studio,cases,chapter,tick).save(PREVIEW/f'{chapter}.png')
        finally:studio.close()
    d.write(PREVIEW/'indicator_checks.json',proof)


def render():
    assert not OUT.exists(),'Refusing overwrite'
    media=old.media_helper();cases=old.load_cases()
    protected=dict(old.s.read(old.OUT/'video_plan.json')['protected_inputs_sha256'])
    for path in [*old.OUT.rglob('*'),*old.study.OUT.rglob('*'),ROOT/'research/make_rr_severity_video.py']:
        if path.is_file():protected[str(path.relative_to(ROOT))]=d.digest(path)
    for name,sha in protected.items():assert d.digest(project_file(ROOT, name))==sha,name
    OUT.mkdir()
    d.write(OUT/'video_plan.json',dict(size=SIZE,fps=FPS,chapters=CHAPTERS,seconds=42,pose=23000,
        sources={str(p.relative_to(ROOT)):d.digest(p) for p in [ROOT/'research/make_rr_severity_visual_video.py',ROOT/'tests/test_make_rr_severity_visual_video.py']},
        protected_inputs_sha256=protected,method='Saved qpos/qvel; mj_forward rendering only. All integration entry points blocked.',
        hierarchy='White layout. Robot and world-space goal first; synchronized clock second; short strength and hold labels third.',
        simulation=False,training=False,final_seeds_used=False))
    storyboard='''# Visual storyboard

- 0–20 s: Healthy and RR75, synchronized at 1x. Identical fixed camera covers the full 2 m mission. Small strength labels change only at the recorded fault latch. Amber marks RR calf; checker floor, directional shadows, center path, goal line and 0.15 m ring show displacement.
- 20–30 s: RR75/50/25/10, synchronized at 2x from simulation 0 to 20 s. Same camera for all panels. Healthy/RR10 archived status remains explicit where shown.
- 30–38 s: Healthy/RR75, synchronized replay of simulation 16–20 s at 0.5x. Identical closer camera at the goal. Live hold bars reset whenever position, heading or speed leaves the original acceptance bounds. The last camera sample is 19.98 s; the result screen uses the exact saved 20 s state.
- 38–42 s: five exact final poses, five outcome dots per condition from all five evaluated poses. No paragraphs or table. This is the only aggregate section.

White layout and Latin Modern Roman are retained. No decorative explanation slides. Green is the goal/hold reference, amber identifies the fault. Every chapter is labelled with playback speed or aggregate scope. The green clock segment marks simulation 18–20 s; it is not measured hold. A complete hold bar alone does not certify every mission criterion. Documentary text stays in README.
'''
    with (OUT/'storyboard.md').open('x',encoding='utf-8') as h:h.write(storyboard)
    with forbidden():
        proof=validate_indicators(cases);d.write(OUT/'indicator_verification.json',proof)
        studio=VisualStudio(media)
        writer=media.imageio_ffmpeg.write_frames(str(OUT/'rr_severity_visual.mp4'),SIZE,fps=FPS,
            codec='libx264',quality=7,macro_block_size=1,ffmpeg_log_level='error',
            output_params=['-an','-movflags','+faststart','-threads','2'])
        writer.send(None);count=0;indices=[]
        try:
            for chapter,seconds in CHAPTERS:
                static=screen(media,studio,cases,chapter,10000) if chapter=='results' else None
                for frame in range(seconds*FPS):
                    tick=frame_tick(chapter,frame)
                    im=static if static is not None else screen(media,studio,cases,chapter,tick)
                    writer.send(np.asarray(im));count+=1
                    indices.append(dict(frame=count-1,chapter=chapter,tick=tick))
                    if frame in [0,seconds*FPS-1]:im.save(OUT/f'{chapter}_{frame}.png')
                    if frame%125==0:print(f'{chapter}: {frame}/{seconds*FPS}',flush=True)
        finally:writer.close();studio.close()
    assert count==1050
    d.write(OUT/'frame_indices.json',indices)
    d.write(OUT/'render_verification.json',dict(frames=count,rendered_poses=studio.drawn,
        qpos_qvel_match_trace=True,physics_arrays_unchanged=True,floor_friction_unchanged=True,
        integration_steps=0,policy_predictions=0,mujoco_version=mujoco.__version__,checker='Original model texture/material restored in rendering copy',
        shadows=True,camera_shared_within_chapter=True))
    verify()


def verify():
    media=old.media_helper();path=OUT/'rr_severity_visual.mp4';exe=media.imageio_ffmpeg.get_ffmpeg_exe()
    count,seconds=media.imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count==1050 and abs(seconds-42)<.01
    reader=media.imageio_ffmpeg.read_frames(str(path));meta=next(reader);reader.close()
    assert tuple(meta['size'])==SIZE and meta['fps']==FPS
    subprocess.run([exe,'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,capture_output=True)
    sheet=Image.new('RGB',(1920,1080),BG)
    for i,t in enumerate([0,9,19.96,22,29.96,30,35,37.96,41.96]):
        target=OUT/f'decoded_{i}.png';assert not target.exists()
        subprocess.run([exe,'-v','error','-ss',str(t),'-i',str(path),'-frames:v','1',str(target)],check=True,capture_output=True)
        with Image.open(target) as im:sheet.paste(im.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(OUT/'contact_sheet.png')
    p=d.read(OUT/'video_plan.json')
    for name,sha in p['sources'].items():assert d.digest(project_file(ROOT, name))==sha,name
    for name,sha in p['protected_inputs_sha256'].items():assert d.digest(project_file(ROOT, name))==sha,name
    d.write(OUT/'verification.json',dict(passed=True,frames=count,seconds=seconds,size=SIZE,fps=FPS,
        all_frames_decoded=True,protected_files_unchanged=len(p['protected_inputs_sha256']),
        indicator_states_checked=50005,new_simulation=False,training=False,final_seeds_used=False,
        video_sha256=d.digest(path),source_video_sha256=d.digest(old.OUT/'rr_severity_comparison.mp4')))
    with (OUT/'README.md').open('x',encoding='utf-8') as h:
        h.write('# RR severity visual revision\n\n42 s, 1920×1080, 25 fps, silent. Exact saved pose 23000 under five conditions. Main pair 1x, four strength levels 2x, final 16–20 s replay 0.5x, followed by all-five-pose outcome dots. Healthy and RR10 are archived controls. RR75 still fails the mission.\n\nThe original MuJoCo checker floor texture is restored on a rendering copy. Shadows, shared fixed cameras, RR knee highlight, a goal line at 2 m and the 0.15 m goal ring replace explanation slides. Clock shading indicates the last 2 s; hold bars use position, heading and speed together and reset on violations. A full hold bar alone does not certify the complete mission.\n\nNo integration or policy inference is run. Integration entry points are blocked during rendering. qpos/qvel are checked exactly on every drawn pose. Physical model arrays and floor friction are unchanged by rendering. The rendered body geometry comes from the same existing model; this is visualization, not a new dynamics replay. Source trace hashes, full-resolution indicators and old artifacts are checked.\n\nLimits: motion shows one development pose; only closing dots aggregate five poses. Timing is quantized to displayed frames. No new seeds, evaluation episodes, training, hardware or generalization checks. Inspect contact_sheet.png and visual_review.json for rendered-frame review.\n')
    print(f'Verified: {path}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preview','render'])
    {'preview':preview,'render':render}[parser.parse_args().stage]()
