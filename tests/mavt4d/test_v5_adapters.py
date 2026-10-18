import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mavt4d.object_state.v5.adapters.manifest import build_dl3dv, arkit_split, safe_member, build_manifest, build_arkit, build_co3d, build_epic, co3d_camera
from mavt4d.object_state.v5.natural import NaturalSources, mixture_counts


class AdapterTests(unittest.TestCase):
    def fixture(self, root):
        p = root / 'scene.zip'
        image = io.BytesIO(); Image.fromarray(np.ones((8, 12, 3), dtype=np.uint8) * 127).save(image, format='PNG')
        frames = [{'file_path': f'images/frame_{i:05d}.png', 'transform_matrix': np.eye(4).tolist()} for i in range(4)]
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr('s/transforms.json', json.dumps({'w': 24, 'h': 16, 'fl_x': 20, 'fl_y': 20, 'cx': 12, 'cy': 8, 'frames': frames}))
            for i in range(4): z.writestr(f's/images_8/frame_{i:05d}.png', image.getvalue())
        return p

    def test_dl3dv_pose_only_and_real_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive = self.fixture(root)
            row = build_dl3dv(root, archive)
            self.assertFalse(row['time_valid'])
            self.assertEqual(row['depth_provenance'], 'missing')
            self.assertEqual(row['frames'][0]['camera_c2w'][1][1], -1)
            row['split'] = 'train'
            manifest = root/'manifest-v1.jsonl'; manifest.write_text(json.dumps(row)+'\n')
            ds = NaturalSources({'root': d, 'dl3dv_manifest': manifest.name, 'mixture': {'dl3dv': 1.}, 'height': 8, 'width': 12})
            clip = ds.draw('dl3dv', 3)
            self.assertEqual(tuple(clip['rgb'].shape), (4, 1, 3, 8, 12))
            self.assertTrue(clip['camera_valid'].all())
            self.assertNotIn('depth', clip)
            self.assertEqual(clip['plan']['future'], [])
            self.assertAlmostEqual(float(clip['intrinsics'][0,0,0,0]), 10.)
            self.assertEqual(ds.inventory()['mixture'], {'dl3dv': 1.})

    def test_arkit_overlap_and_unknown_visit_quarantine(self):
        for video in ['41159521', '41159526', '40753686']:
            self.assertEqual(arkit_split(video, 'Training', None), 'quarantine')
        self.assertEqual(arkit_split('41159519', 'Validation', '381879'), 'val')

    def test_reject_traversal_and_incomplete(self):
        for path in ['../x', '/x', 'x/../../a']:
            with self.assertRaises(ValueError): safe_member(path)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=self.fixture(root); p.rename(root/'scene.zip.part')
            result=build_manifest(root, 'dl3dv', root/'out-v1.jsonl')
            self.assertEqual(result['usable_records'], 0)

    def test_arkit_metric_disjoint_depth(self):
        from mavt4d.object_state.v5.adapters.decode import decode_record
        import random
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'123.zip'; image=io.BytesIO(); depth=io.BytesIO()
            Image.fromarray(np.ones((8,12,3),dtype=np.uint8)*100).save(image,format='PNG')
            Image.fromarray(np.ones((8,12),dtype=np.uint16)*2000).save(depth,format='PNG')
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('123/lowres_wide.traj','0.0 0 0 0 0 0 0\n1.0 0 0 0 0 0 0')
                for t in ['0.000','1.000']:
                    z.writestr(f'123/lowres_wide/123_{t}.png',image.getvalue())
                    z.writestr(f'123/lowres_depth/123_{t}.png',depth.getvalue())
                    z.writestr(f'123/lowres_wide_intrinsics/123_{t}.pincam','12 8 10 10 6 4')
            row=build_arkit(root,archive,{'123':{'visit_id':'42','fold':'Training'}})
            clip=decode_record(root,row,random.Random(0),8,12)
            self.assertTrue(clip['metric_valid'].all())
            self.assertFalse(((clip['depth']>0)&(clip['input_depth']>0)).any())
            self.assertEqual(float(clip['depth'].max()),2.)

    def test_co3d_conversion_and_sequence_group(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'apple_000.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('apple/seq/images/1.jpg',b'x')
            view={'R':np.eye(3).tolist(),'T':[0,0,0],'focal_length':[2,2], 'principal_point':[0,0], 'intrinsics_format':'ndc_isotropic'}
            ann=[{'sequence_name':'seq','frame_number':1,'frame_timestamp':.25,'image':{'path':'apple/seq/images/1.jpg','size':[8,12]},'viewpoint':view}]
            row=build_co3d(root,archive,ann)[0]
            self.assertEqual(row['group'],'co3d/seq')
            self.assertFalse(row['metric_valid'])
            self.assertEqual(row['frames'][0]['intrinsics'][0][0],8.)
            self.assertEqual(row['frames'][0]['camera_c2w'][0][0],-1.)

    def test_epic_split_original_recording_not_container(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'bucket.zip'
            clips=['epic_kitchen/P01/P01_01/video1/rgb','epic_kitchen/P01/P01_01/video2/rgb']
            with zipfile.ZipFile(archive,'w') as z:
                for clip in clips: z.writestr(clip+'/00000.png',b'x')
            rows=build_epic(root,archive,{clip:{'FPS':'30','Annotation Path':'annotation'} for clip in clips})
            self.assertEqual(rows[0]['group'],rows[1]['group'])
            self.assertEqual(rows[0]['split'],rows[1]['split'])
            self.assertEqual(rows[0]['depth_provenance'],'missing')

    def test_frozen_archive_replacement_rejected(self):
        from mavt4d.object_state.v5.adapters.decode import decode_record
        import os, random
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=self.fixture(root); row=build_dl3dv(root,archive)
            os.utime(archive,ns=(archive.stat().st_atime_ns,archive.stat().st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError,'frozen archive'):
                decode_record(root,row,random.Random(0),8,12)

    def test_long_30fps_archive_retains_all_six_future_horizons(self):
        from mavt4d.object_state.v5.adapters.decode import decode_record
        from mavt4d.object_state.v5.adapters.manifest import record
        from mavt4d.object_state.v5.natural import sample_plan, observation_schedule
        import random
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'video.zip'; image=io.BytesIO()
            Image.fromarray(np.ones((8,12,3),dtype=np.uint8)).save(image,format='PNG')
            frames=[{'rgb':f'{i}.png','time':i/30} for i in range(600)]
            with zipfile.ZipFile(archive,'w') as z:
                for f in frames: z.writestr(f['rgb'],image.getvalue())
            row=record('omniworld_epic','recording',archive,root,frames)
            count=2*(len(observation_schedule('G',17))+6)
            clip=decode_record(root,row,random.Random(5),8,12,max_frames=count)
            plan=sample_plan(clip['times'],'G',seed=17,horizon=6)
            self.assertEqual(len(plan['future']),6)
            self.assertLessEqual(len(clip['times']),count)
            for time in clip['times']:
                self.assertAlmostEqual(float(time)*30,round(float(time)*30),places=8)

    def test_openvid_30fps_grid_retains_six_horizons(self):
        import cv2, random
        from unittest.mock import patch
        from mavt4d.object_state.v5.natural import sample_plan, observation_schedule
        class Capture:
            index=0
            def get(self, prop):
                return {cv2.CAP_PROP_FPS:30,cv2.CAP_PROP_FRAME_COUNT:600,
                        cv2.CAP_PROP_POS_MSEC:self.index/30*1000}[prop]
            def set(self, prop, index): self.index=index
            def read(self): return True,np.ones((8,12,3),dtype=np.uint8)
            def release(self): pass
        ds=NaturalSources.__new__(NaturalSources);ds.height=8;ds.width=12
        with patch('cv2.VideoCapture',return_value=Capture()):
            clip=ds._video(Path('unused.mp4'),random.Random(5),'G',6)
        plan=sample_plan(clip['times'],'G',seed=17,horizon=6)
        self.assertEqual(len(plan['future']),6)
        self.assertLessEqual(len(clip['times']),2*(len(observation_schedule('G',17))+6))

    def test_nuscenes_fingerprint_pins_calibration_and_resolved_split(self):
        from mavt4d.object_state.v5.natural import nuscenes_metadata_fingerprint
        from mavt4d.data.nuscenes import _TABLES
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); tables=root/'v1.0-trainval';tables.mkdir()
            for name in _TABLES: (tables/(name+'.json')).write_text('[]')
            baseline=nuscenes_metadata_fingerprint(root,'v1.0-trainval',['scene-a'])
            (tables/'calibrated_sensor.json').write_text('[{}]')
            self.assertNotEqual(baseline,nuscenes_metadata_fingerprint(root,'v1.0-trainval',['scene-a']))
            (tables/'calibrated_sensor.json').write_text('[]')
            self.assertNotEqual(baseline,nuscenes_metadata_fingerprint(root,'v1.0-trainval',['scene-b']))

    def test_new_mixture(self):
        self.assertEqual(mixture_counts(256, {'arkitscenes': .5, 'co3d': .5}), {'arkitscenes':128,'co3d':128})

if __name__ == '__main__': unittest.main()
