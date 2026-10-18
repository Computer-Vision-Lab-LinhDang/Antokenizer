"""Build immutable scene manifests from complete publisher archives.

No extraction, inferred FPS, or synthetic depth. Camera coordinates are converted
into OpenCV (+x right,+y down,+z forward); all paths stay inside approved root.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

import numpy as np

SCHEMA = 'mavt-natural-v5-v1'
SOURCES = ('dl3dv', 'arkitscenes', 'co3d', 'omniworld_epic')
MAX_MEMBER = 256 * 1024 * 1024


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('unsafe archive member')
    return str(p)


def safe_path(root, name):
    safe_member(str(name))
    p = (Path(root) / name).resolve()
    if not p.is_relative_to(Path(root).resolve()) or not p.is_file() or p.name.endswith('.part'):
        raise ValueError('missing, incomplete, or out-of-root media')
    return p


class Store:
    """Archive handles live for one draw; tar reads are streaming, memory bounded."""
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None

    def names(self):
        if self.zip:
            for item in self.zip.infolist():
                safe_member(item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('archive symlink refused')
                if not item.is_dir(): yield item.filename
        else:
            with tarfile.open(self.path, 'r|*') as stream:
                for item in stream:
                    safe_member(item.name)
                    if item.issym() or item.islnk(): raise ValueError('archive links refused')
                    if item.isfile(): yield item.name

    def read_many(self, names):
        wanted = set(map(safe_member, names)); output = {}
        if self.zip:
            for name in wanted:
                item = self.zip.getinfo(name)
                if item.file_size > MAX_MEMBER: raise ValueError('member exceeds bounded decode limit')
                output[name] = self.zip.read(item)
        else:
            with tarfile.open(self.path, 'r|*') as stream:
                for item in stream:
                    if item.name in wanted:
                        if not item.isfile() or item.size > MAX_MEMBER: raise ValueError('unsafe member')
                        output[item.name] = stream.extractfile(item).read()
                        if len(output) == len(wanted): break
        if wanted != output.keys(): raise ValueError('missing archive member')
        return output

    def read(self, name): return self.read_many([name])[name]
    def close(self):
        if self.zip: self.zip.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def group_split(group):
    from mavt.data.splits import bucket
    if bucket(group, 'mavt-split-v1') < .02: return 'val'
    return 'dev' if bucket(group, 'mavt-v5-internal-dev-v1') < .02 else 'train'


def arkit_split(video, fold, visit):
    if fold == 'Validation': return 'val'
    if fold != 'Training': raise ValueError('unknown ARKit official fold')
    if str(video) in {'41159521', '41159526', '40753686'} or str(visit) == '381879':
        return 'quarantine'
    # Missing visits are quarantined: no proof of cross-recording separation.
    if visit in (None, '', 'NA', 'nan'): return 'quarantine'
    from mavt.data.splits import bucket
    return 'dev' if bucket(str(visit), 'mavt-v5-internal-dev-v1') < .02 else 'train'


def record(source, group, archive, root, frames, **extras):
    frames.sort(key=lambda x: x['time'])
    if not frames: raise ValueError('empty scene')
    if any(b['time'] <= a['time'] for a,b in zip(frames,frames[1:])): raise ValueError('duplicate/nonmonotonic timestamps')
    return dict(schema=SCHEMA, source=source, group=f'{source}/{group}',
                split=group_split(f'{source}/{group}'), archive=str(archive.relative_to(root)),
                archive_size=archive.stat().st_size, archive_mtime_ns=archive.stat().st_mtime_ns, frames=frames,
                depth_provenance='missing', metric_valid=False, time_valid=True,
                timestamp_provenance='source', **extras)


def build_dl3dv(root, archive):
    with Store(archive) as st:
        names = list(st.names()); meta = next((n for n in names if n.endswith('transforms.json')), None)
        if meta is None: raise ValueError('missing transforms')
        data = json.loads(st.read(meta)); parent = str(PurePosixPath(meta).parent)
        images = {PurePosixPath(n).name: n for n in names if '/images_8/' in n or '/images/' in n}
        frames=[]; timed=all('time' in f for f in data['frames'])
        for i,f in enumerate(data['frames']):
            name=images.get(PurePosixPath(f['file_path']).name)
            if name is None: continue
            values={**data,**f}; pose=np.asarray(f['transform_matrix'],dtype=float) @ np.diag([1,-1,-1,1])
            model = values.get('camera_model', 'OPENCV')
            if model not in ('OPENCV', 'PINHOLE', 'SIMPLE_PINHOLE'):
                raise ValueError('unsupported lens model; rectification required')
            frame=dict(rgb=name,time=float(f['time']) if timed else float(i))
            frame.update(intrinsics=[[values['fl_x'],0,values['cx']],[0,values['fl_y'],values['cy']],[0,0,1]],
                         calibration_size=[values['h'],values['w']],camera_c2w=pose.tolist(),
                         distortion_opencv=[float(values.get(k,0)) for k in ('k1','k2','p1','p2','k3')])
            frames.append(frame)
        out=record('dl3dv',archive.stem,archive,root,frames)
        out.update(time_valid=timed,timestamp_provenance='source_time' if timed else 'frame_order_only',camera_provenance='reconstructed_colmap')
        return out


def build_arkit(root, archive, metadata):
    video=archive.stem; m=metadata[video]
    with Store(archive) as st:
        names=list(st.names()); poses={}; cameras={}; frames=[]
        traj=next((n for n in names if n.endswith('lowres_wide.traj')),None)
        if not traj: raise ValueError('missing ARKit trajectory')
        import cv2
        for line in st.read(traj).decode().splitlines():
            v=list(map(float,line.split()))
            if len(v)!=7: raise ValueError('invalid trajectory')
            w2c=np.eye(4); w2c[:3,:3]=cv2.Rodrigues(np.array(v[1:4]))[0]; w2c[:3,3]=v[4:]
            poses[round(v[0],3)]=np.linalg.inv(w2c).tolist()
        pin=[n for n in names if n.endswith('.pincam')]
        for n,b in st.read_many(pin).items():
            w,h,fx,fy,cx,cy=map(float,b.split()); stamp=float(PurePosixPath(n).stem.split('_')[-1])
            cameras[stamp]=([[fx,0,cx],[0,fy,cy],[0,0,1]],[h,w])
        nameset=set(names)
        for name in names:
            if '/lowres_wide/' not in name or not name.endswith('.png'): continue
            t=float(PurePosixPath(name).stem.split('_')[-1]); pt=min(poses,key=lambda v:abs(v-t)); kt=min(cameras,key=lambda v:abs(v-t))
            if abs(pt-t)>.005 or abs(kt-t)>.002: continue
            k,size=cameras[kt]; f=dict(rgb=name,time=t,intrinsics=k,calibration_size=size,camera_c2w=poses[pt])
            dep=name.replace('/lowres_wide/','/lowres_depth/')
            if dep in nameset: f.update(depth=dep,depth_scale=.001,depth_encoding='uint16')
            frames.append(f)
        out=record('arkitscenes',str(m.get('visit_id') or video),archive,root,frames)
        out.update(split=arkit_split(video,m['fold'],m.get('visit_id')),depth_provenance='measured',
                   metric_valid=True,timestamp_provenance='sensor_timestamp',sequence=video,
                   depth_detail='lowres_depth_sensor_mm',camera_provenance='arkit_sensor_pose',
                   static_scene_evidence={'protocol':'arkitscenes-static-capture-v1',
                       'reference':'arxiv:2111.08897','scope':'sequence','sequence':video,
                       'claim':'publisher_capture_assumption_sensor_checked'})
        return out


def co3d_camera(v, size):
    """PyTorch3D row-vector NDC camera -> OpenCV pixel K and column-vector c2w."""
    h,w=size; f=np.asarray(v['focal_length']); p=np.asarray(v['principal_point'])
    fmt=v.get('intrinsics_format','ndc_norm_image_bounds')
    if fmt=='ndc_norm_image_bounds': scale=np.array([w,h])/2
    elif fmt=='ndc_isotropic': scale=np.full(2,min(h,w)/2)
    else: raise ValueError('unsupported CO3D intrinsics convention')
    focal=f*scale; center=np.array([w,h])/2-p*scale
    k=[[focal[0],0,center[0]],[0,focal[1],center[1]],[0,0,1]]
    w2c=np.eye(4); flip=np.diag([-1,-1,1]); w2c[:3,:3]=flip@np.asarray(v['R']).T; w2c[:3,3]=flip@np.asarray(v['T'])
    return k,np.linalg.inv(w2c).tolist()


def build_co3d(root, archive, annotations):
    with Store(archive) as st:
        names=set(st.names()); groups=defaultdict(list)
        for a in annotations:
            name=a['image']['path']
            if name not in names: continue
            f=dict(rgb=name,time=float(a.get('frame_timestamp',a['frame_number'])))
            if a.get('viewpoint'):
                k,c=co3d_camera(a['viewpoint'],a['image']['size']); f.update(intrinsics=k,camera_c2w=c,calibration_size=a['image']['size'])
            dep=a.get('depth') or {}
            if dep.get('path') in names:
                f.update(depth=dep['path'],depth_scale=dep.get('scale_adjustment',1.),depth_encoding='float16_bits')
                if dep.get('mask_path') in names: f['depth_mask']=dep['mask_path']
            groups[a['sequence_name']].append(f)
        rows=[]
        for seq,frames in groups.items():
            out=record('co3d',seq,archive,root,frames)
            timed=all('frame_timestamp' in a for a in annotations if a['sequence_name']==seq)
            out.update(category=PurePosixPath(frames[0]['rgb']).parts[0],depth_provenance='reconstructed',
                       time_valid=timed,timestamp_provenance='source_time' if timed else 'frame_order_only',camera_provenance='reconstructed_sfm')
            rows.append(out)
        return rows


def build_epic(root, archive, metadata):
    groups=defaultdict(list)
    with Store(archive) as st:
        for name in st.names():
            p=PurePosixPath(name); clip=str(p.parent)
            if clip not in metadata or p.suffix!='.png' or not p.stem.isdigit(): continue
            fps=float(metadata[clip]['FPS'])
            if not math.isfinite(fps) or fps<=0: raise ValueError('invalid metadata FPS')
            groups[clip].append(dict(rgb=name,time=int(p.stem)/fps))
    rows=[]
    for clip,frames in groups.items():
        parts=PurePosixPath(clip).parts
        if len(parts)<4: raise ValueError('EPIC original recording missing')
        out=record('omniworld_epic','/'.join(parts[:3]),archive,root,frames)
        out.update(sequence=clip,timestamp_provenance='publisher_metadata_fps',
                   annotation_reference=metadata[clip].get('Annotation Path'),
                   annotation_status='not_loaded_no_geometry_claim')
        rows.append(out)
    return rows


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''): h.update(data)
    return h.hexdigest()


def build_manifest(root, source, output, *, metadata=None, archives=None):
    """Freeze only existing complete, parsed archives. Output is exclusive/versioned.

    metadata: ARKit or EPIC CSV; CO3D frame_annotations.jgz. For sharded CO3D
    call per category with its annotation file, then combine via source manifest.
    """
    root=Path(root).resolve(); output=Path(output)
    if source not in SOURCES: raise ValueError('unsupported source')
    if output.exists() or output.with_suffix(output.suffix+'.audit.json').exists(): raise FileExistsError('versioned output exists')
    meta={}
    if metadata:
        metadata=Path(metadata)
        if source in ('arkitscenes','omniworld_epic'):
            with metadata.open() as f:
                key='video_id' if source=='arkitscenes' else 'Video Path'
                meta={r[key]:r for r in csv.DictReader(f)}
        elif source=='co3d':
            with gzip.open(metadata,'rt') as f: meta=json.load(f)
    archive_paths = sorted(archives) if archives is not None else sorted(p for p in root.rglob('*') if p.is_file() and (p.suffix=='.zip' or p.name.endswith(('.tar.gz','.tgz'))))
    audit={'schema':SCHEMA,'source':source,'usable_records':0,'rejected':[],'archives':[],'metadata_sha256':sha256(metadata) if metadata else None}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        for archive in archive_paths:
            archive=Path(archive).resolve()
            if archive.name.endswith('.part'): continue
            if not archive.is_relative_to(root): raise ValueError('archive outside root')
            try:
                if source=='dl3dv': rows=[build_dl3dv(root,archive)]
                elif source=='arkitscenes': rows=[build_arkit(root,archive,meta)]
                elif source=='co3d': rows=build_co3d(root,archive,meta)
                else: rows=build_epic(root,archive,meta)
                if not rows: raise ValueError('no usable records')
                digest=sha256(archive)
                audit['archives'].append({'path':str(archive.relative_to(root)),'sha256':digest,'size':archive.stat().st_size})
                for row in rows:
                    row['archive_sha256']=digest
                    stream.write(json.dumps(row,sort_keys=True)+'\n'); audit['usable_records']+=1
            except (ValueError,KeyError,StopIteration,zipfile.BadZipFile,tarfile.TarError,OSError) as e:
                audit['rejected'].append({'archive':str(archive.relative_to(root)),'reason':type(e).__name__+': '+str(e)})
    audit['manifest_sha256']=sha256(output)
    with output.with_suffix(output.suffix+'.audit.json').open('x') as f: json.dump(audit,f,indent=2)
    return audit
