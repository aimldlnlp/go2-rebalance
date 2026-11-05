"""Presentation-only revision; reuse all verified motion, data and encoding."""
from project_paths import project_file
import argparse
from pathlib import Path
import shutil
import os
import numpy as np
from PIL import Image,ImageDraw
import make_walking_progress_data_video as base

ROOT=base.ROOT
OUT=ROOT/'results/walking_progress_data_video_minimal'
CURRENT=base.OUT/'walking_progress_data.mp4'
BACKUP=base.OUT/'walking_progress_data_previous.mp4'
PREVIEW=ROOT/'work/walking_progress_video_minimal_preview'
original_text=base.text


def compact_text(media,im,value,x,y,size=25,color=base.INK):
    value={'Hip / upper leg / knee':'','All four weakened knees':'20 runs',
           'Five starting poses each':'','Goal area':'Goal','Time (s)':'s',
           'More progress':'More reward'}.get(value,value)
    if value:
        original_text(media,im,value,x,y,size,color)


def text(media,im,value,x,y,size=25,color=base.INK):
    original_text(media,im,value,x,y,size,color)


def scale(media,im,x,y,width=230):
    draw=ImageDraw.Draw(im)
    low=np.array([239,241,237]);high=np.array([83,108,67])
    for i in range(width):
        color=tuple(np.rint(low+(high-low)*i/(width-1)).astype(int))
        draw.line((x+i,y,x+i,y+7),fill=color)
    text(media,im,'0',x,y+14,18,base.MUTED)
    text(media,im,'5 Nm',x+width-46,y+14,18,base.MUTED)


def screen(media,studio,data,chapter,frame):
    im=Image.new('RGB',base.SIZE,'white');draw=ImageDraw.Draw(im)
    now=base.ticks(chapter,frame,data)
    if chapter=='healthy':
        c=data['healthy'];tick=now['healthy']
        text(media,im,'Healthy · Archive',40,26,32,base.COLORS['healthy'])
        im.paste(studio.frame(c,tick,(1240,690),base.COLORS['healthy']),(40,90))
        base.space_time(media,im,{'healthy':c},{'healthy':tick},(1310,165,560,480))
        base.hold_ring(media,im,c['hold'][tick],1400,710)
        text(media,im,'Hold',1525,746,28)
        text(media,im,'2 m / 20 s',40,870,38)
        text(media,im,'Goal 0.15 m',40,935,25,base.MUTED)
        text(media,im,f'{tick*.002:.2f} s · 4×',40,1025,25,base.MUTED)
        return im
    if chapter=='comparison':
        main={name:data['cases'][name][23000] for name in base.FOLDERS}
        for name,x in [('original',40),('progress',1290)]:
            c=main[name];tick=now[name]
            text(media,im,'Original' if name=='original' else 'More reward',x,26,33,base.COLORS[name])
            im.paste(studio.frame(c,tick,(590,360),base.COLORS[name]),(x,90))
            strength=10 if tick>=c['row']['fault_onset_tick'] else 100
            text(media,im,f'Knee {strength}%',x,467,25,base.MUTED)
        base.space_time(media,im,main,now,(650,100,600,365))
        base.curve(media,im,data,now,'error',(40,560,590,200),'To goal (m)')
        base.curve(media,im,data,now,'speed',(40,790,590,200),'Forward speed (m/s)')
        for name,x in [('original',665),('progress',1290)]:
            base.heatmap(media,im,data['envelopes'][name]['binned'],now[name],
                (x,560,570,400),'Original' if name=='original' else 'More reward',25)
        text(media,im,'Motor range',665,1000,24)
        scale(media,im,850,1002,240)
        text(media,im,f'{now["original"]*.002:05.2f} s · 1×',40,1030,24,base.MUTED)
        return im
    if chapter=='detail':
        for name,x in [('original',40),('progress',1000)]:
            c=data['cases'][name][23000];tick=now[name]
            text(media,im,'Original' if name=='original' else 'More reward',x,26,33,base.COLORS[name])
            im.paste(studio.frame(c,tick,(880,380),base.COLORS[name]),(x,90))
            base.heatmap(media,im,data['envelopes'][name]['binned'],tick,
                (x,540,880,340),'Motor range',28)
            torque=min(tick,9999);e=data['envelopes'][name]
            span=e['high'][torque,11]-e['low'][torque,11]
            text(media,im,f'Knee {span:.3f} Nm',x,905,33,base.COLORS[name])
            elapsed=(tick-c['row']['fault_onset_tick'])*.002
            text(media,im,f'+{elapsed:.2f} s',x,1006,24,base.MUTED)
        scale(media,im,1550,920,230)
        text(media,im,'0.5×',1790,1025,25,base.MUTED)
        return im
    for name,y in [('original',40),('progress',500)]:
        text(media,im,'Original' if name=='original' else 'More reward',40,y,31,base.COLORS[name])
        im.paste(studio.frame(data['cases'][name][23000],10000,(690,335),base.COLORS[name]),(40,y+55))
    base.outcomes(media,im,data,(830,40,1020,850),True)
    draw.ellipse((835,944,848,957),fill='#a35317')
    text(media,im,'Fall',860,938,23,base.MUTED)
    draw.ellipse((980,944,993,957),fill='#c1c6bd')
    text(media,im,'Upright',1005,938,23,base.MUTED)
    text(media,im,'Saved simulation',40,1026,23,base.MUTED)
    return im


def wire():
    base.OUT=OUT
    base.PREVIEW=PREVIEW
    base.text=compact_text
    base.screen=screen


def prepare():
    assert CURRENT.exists() and not BACKUP.exists()
    assert 'OK' in (ROOT/'work/progress_video_minimal_tests.log').read_text(encoding='utf-8')
    old=base.s.read(CURRENT.parent/'verification.json')
    assert base.s.digest(CURRENT)==old['video_sha256']
    base.prepare()
    shutil.copyfile(ROOT/'work/progress_video_minimal_tests.log',OUT/'minimal_tests.log')
    base.s.write(OUT/'revision_plan.json',dict(presentation_only=True,
        replace_target=str(CURRENT.relative_to(ROOT)),original_sha256=base.s.digest(CURRENT),
        backup_target=str(BACKUP.relative_to(ROOT)),
        unchanged='All motion, data, frame count and source tick indices; no training or evaluation',
        cuts='Remove explanatory sentences, repeated titles and in-motion outcome matrix; shorten method labels; outcomes only in closing chapter',
        retained='Units, time/playback, archive label, strength, anatomical group labels, heatmap scale, mission/fall outcomes and recorded-simulation label',
        detail_clock='Exact elapsed saved time after physical fault latch for each arm',
        minimal_tests_sha256=base.s.digest(OUT/'minimal_tests.log')))


def replace():
    plan=base.s.read(OUT/'revision_plan.json')
    proof=base.s.read(OUT/'verification.json');assert proof['passed']
    stage=OUT/'walking_progress_data.mp4'
    assert base.s.digest(stage)==proof['video_sha256']
    assert base.s.digest(CURRENT)==plan['original_sha256'] and not BACKUP.exists()
    old_indices=base.s.read(CURRENT.parent/'frame_indices.json')
    new_indices=base.s.read(OUT/'frame_indices.json')
    assert old_indices==new_indices,'Presentation must preserve all source ticks'
    assert base.s.digest(OUT/'minimal_tests.log')==plan['minimal_tests_sha256']
    # Save and verify the original before the single authorized replacement.
    shutil.copyfile(CURRENT,BACKUP)
    assert base.s.digest(BACKUP)==plan['original_sha256']
    pending=CURRENT.with_name('walking_progress_data.replacement.mp4')
    assert not pending.exists()
    shutil.copyfile(stage,pending)
    assert base.s.digest(pending)==proof['video_sha256']
    os.replace(pending,CURRENT)
    assert base.s.digest(CURRENT)==proof['video_sha256']
    full=base.s.read(OUT/'video_plan.json')
    for name,sha in full['protected_inputs'].items():
        if project_file(ROOT, name)!=CURRENT:assert base.s.digest(project_file(ROOT, name))==sha,name
    result=dict(passed=True,current_video=str(CURRENT),current_sha256=proof['video_sha256'],
        original_backup=str(BACKUP),original_sha256=plan['original_sha256'],
        old_video_metadata_preserved='Original sidecars describe the backed-up video; new verification lives beside the minimal revision.',
        source_frame_indices_identical=True,other_protected_inputs_unchanged=len(full['protected_inputs'])-1,
        new_training_or_evaluation=False)
    base.s.write(OUT/'replacement.json',result)
    base.s.write(CURRENT.parent/'current_video_revision.json',dict(current_sha256=proof['video_sha256'],
        verification=str((OUT/'verification.json').relative_to(ROOT)),
        revision=str((OUT/'replacement.json').relative_to(ROOT)),original_backup=BACKUP.name))
    print(CURRENT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preview','prepare','render','replace'])
    args=p.parse_args();wire()
    {'preview':base.preview,'prepare':prepare,'render':base.render,'replace':replace}[args.command]()
