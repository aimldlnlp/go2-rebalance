import unittest
from unittest.mock import patch
from PIL import Image
import make_walking_progress_video_minimal as v


class BlankStudio:
    def frame(self,case,tick,size,color):
        return Image.new('RGB',size,'#dddddd')


class Tests(unittest.TestCase):
    def test_text_reduction_and_required_meaning(self):
        data=v.base.load();media=v.base.media_old.media_helper()
        oldscreen=v.base.screen
        words={};captions={}
        for chapter,frame in [('healthy',120),('comparison',375),('detail',150),('results',0)]:
            current=[]
            def record(media,im,value,*args,**kwargs):current.append(value)
            with patch.object(v.base,'text',record):
                oldscreen(media,BlankStudio(),data,chapter,frame)
            old=len(' '.join(current).split())
            current=[]
            def compact(media,im,value,*args,**kwargs):
                value={'Hip / upper leg / knee':'','All four weakened knees':'20 runs',
                    'Five starting poses each':'','Goal area':'Goal','Time (s)':'s',
                    'More progress':'More reward'}.get(value,value)
                if value:current.append(value)
            with patch.object(v,'text',record),patch.object(v.base,'text',compact):
                v.screen(media,BlankStudio(),data,chapter,frame)
            new=len(' '.join(current).split())
            self.assertLess(new,old*.72,(chapter,old,new))
            words[chapter]=(old,new);captions[chapter]=current
        self.assertIn('Healthy · Archive',captions['healthy'])
        self.assertIn('Knee 10%',captions['comparison'])
        self.assertIn('Completed',captions['results'])
        self.assertIn('Falls',captions['results'])
        self.assertEqual(captions['results'].count('0/20'),2)
        self.assertIn('7/20',captions['results']);self.assertIn('6/20',captions['results'])
        self.assertIn('Saved simulation',captions['results'])
        print('Word counts (before,after):',words,flush=True)


if __name__=='__main__':unittest.main()
