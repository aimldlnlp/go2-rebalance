"""Widen the closing views without changing the verified motion chapters."""
import subprocess
import numpy as np
from PIL import Image
import make_rr_severity_visual_video as v


def main():
    out=v.OUT;media=v.old.media_helper()
    source=out/'rr_severity_visual.mp4';target=out/'rr_severity_visual_final.mp4'
    assert not target.exists()
    old_proof=v.d.read(out/'verification.json')
    assert old_proof['passed'] and v.d.digest(source)==old_proof['video_sha256']
    cases=v.old.load_cases()
    with v.forbidden():
        studio=v.VisualStudio(media)
        studio.base.model.vis.global_.fovy=60
        try:ending=v.screen(media,studio,cases,'results',10000)
        finally:studio.close()
    ending.save(out/'results_uncropped.png')
    reader=media.imageio_ffmpeg.read_frames(str(source));meta=next(reader)
    assert tuple(meta['size'])==v.SIZE and meta['fps']==v.FPS
    writer=media.imageio_ffmpeg.write_frames(str(target),v.SIZE,fps=v.FPS,codec='libx264',quality=7,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-an','-movflags','+faststart','-threads','2'])
    writer.send(None);count=0
    try:
        for i,frame in enumerate(reader):
            writer.send(np.asarray(ending) if i>=950 else frame);count+=1
    finally:reader.close();writer.close()
    assert count==1050
    exe=media.imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe,'-v','error','-xerror','-i',str(target),'-f','null','-'],check=True,capture_output=True)
    frames,seconds=media.imageio_ffmpeg.count_frames_and_secs(str(target))
    assert frames==1050 and abs(seconds-42)<.01
    psnr=subprocess.run([exe,'-hide_banner','-i',str(source),'-i',str(target),'-filter_complex',
        '[0:v]trim=end=38,setpts=PTS-STARTPTS[a];[1:v]trim=end=38,setpts=PTS-STARTPTS[b];[a][b]psnr',
        '-an','-f','null','-'],check=True,capture_output=True,text=True)
    import re
    match=re.search(r'average:([\d.]+)',psnr.stderr);assert match and float(match[1])>35
    sheet=Image.new('RGB',v.SIZE,'white')
    for i,t in enumerate([0,9,19.96,22,29.96,30,35,37.96,41.96]):
        path=out/f'final_decoded_{i}.png';assert not path.exists()
        subprocess.run([exe,'-v','error','-ss',str(t),'-i',str(target),'-frames:v','1',str(path)],check=True,capture_output=True)
        with Image.open(path) as image:sheet.paste(image.resize((640,360)),((i%3)*640,(i//3)*360))
    sheet.save(out/'final_contact_sheet.png')
    v.d.write(out/'final_verification.json',dict(passed=True,frames=frames,seconds=seconds,size=v.SIZE,
        fps=v.FPS,all_frames_decoded=True,motion_chapters_frame_order_and_timing_preserved=True,
        first_38_seconds_reencoding_psnr_db=float(match[1]),closing_camera_fovy=60,
        closing_new_rendered_poses=5,closing_tick=10000,video_sha256=v.d.digest(target),
        source_video_sha256=v.d.digest(source),finalizer_sha256=v.d.digest(__file__),
        original_video_retained=True,new_simulation=False,training=False))
    print(target,flush=True)


if __name__=='__main__':main()
