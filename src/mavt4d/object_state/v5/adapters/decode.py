"""Decode bounded windows from a frozen manifest; depth targets stay held out."""
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .manifest import Store, safe_path


def static_capture_evidence(row):
    """Publisher acquisition assumption, explicitly scoped; never a scene GT claim."""
    evidence = row.get('static_scene_evidence') or {}
    return (row.get('source') == 'arkitscenes' and row.get('depth_provenance') == 'measured'
            and evidence.get('protocol') == 'arkitscenes-static-capture-v1'
            and evidence.get('reference') == 'arxiv:2111.08897'
            and evidence.get('scope') == 'sequence'
            and bool(row.get('sequence')) and evidence.get('sequence') == row.get('sequence')
            and evidence.get('claim') == 'publisher_capture_assumption_sensor_checked')


def decode_record(root, row, rng, height, width, *, max_frames=36):
    frames=row['frames']
    if max_frames < 1 or not frames:
        raise ValueError('positive frame budget and nonempty scene required')
    if row.get('time_valid', True) and len(frames) > 1:
        # Tubelet endpoints are every second frame. A 0.25s sampling grid
        # therefore leaves one real endpoint per 0.5s rollout horizon. Read only
        # nearest actual source frames; never synthesize timestamps or duplicates.
        source_times=np.asarray([frame['time'] for frame in frames],dtype=np.float64)
        if not np.isfinite(source_times).all() or not (np.diff(source_times)>0).all():
            raise ValueError('source timestamps must strictly increase')
        span=(max_frames-1)*.25
        last_start=max(0,int(np.searchsorted(source_times,source_times[-1]-span,side='right'))-1)
        start=rng.randint(0,last_start)
        desired=source_times[start]+np.arange(max_frames)*.25
        selected=[]
        for time in desired:
            if time > source_times[-1]+1e-9: break
            right=int(np.searchsorted(source_times,time))
            candidates=[index for index in (right-1,right) if start<=index<len(frames)]
            index=min(candidates,key=lambda i:abs(source_times[i]-time))
            if not selected or index>selected[-1]: selected.append(index)
        chosen=[frames[index] for index in selected]
    else:
        start=rng.randint(0,max(0,len(frames)-max_frames))
        chosen=frames[start:start+max_frames]
    archive=safe_path(root,row['archive'])
    if archive.stat().st_size != row['archive_size'] or archive.stat().st_mtime_ns != row.get('archive_mtime_ns', archive.stat().st_mtime_ns): raise ValueError('frozen archive changed')
    names=[f[key] for f in chosen for key in ('rgb','depth','depth_mask') if key in f]
    with Store(archive) as st: payload=st.read_many(names)
    images=[]; image_valid=[]; depths=[]; inputs=[]; cameras=[]; intrinsics=[]; camera_valid=[]
    for frame in chosen:
        with Image.open(io.BytesIO(payload[frame['rgb']])) as im:
            pixels=np.asarray(im.convert('RGB')); ih,iw=pixels.shape[:2]
        valid=np.ones((ih,iw),dtype=np.uint8)
        if any(frame.get('distortion_opencv', [])):
            import cv2
            k=np.asarray(frame['intrinsics'],dtype=np.float64).copy(); ch,cw=frame['calibration_size']
            k[0]*=iw/cw; k[1]*=ih/ch
            maps=cv2.initUndistortRectifyMap(k,np.asarray(frame['distortion_opencv']),None,k,(iw,ih),cv2.CV_32FC1)
            pixels=cv2.remap(pixels,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            valid=cv2.remap(valid,*maps,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        images.append(torch.from_numpy(np.asarray(Image.fromarray(pixels).resize((width,height),Image.Resampling.BILINEAR)).copy()).permute(2,0,1).float()/255)
        image_valid.append(torch.from_numpy(np.asarray(Image.fromarray(valid).resize((width,height),Image.Resampling.NEAREST)).copy()).bool()[None])
        has_camera='intrinsics' in frame and 'camera_c2w' in frame
        camera_valid.append(has_camera)
        if has_camera:
            k=torch.tensor(frame['intrinsics'],dtype=torch.float32); h,w=frame['calibration_size']
            k[0]*=width/w; k[1]*=height/h
            intrinsics.append(k); cameras.append(torch.tensor(frame['camera_c2w'],dtype=torch.float32))
        else:
            intrinsics.append(torch.eye(3)); cameras.append(torch.eye(4))
        target=torch.zeros(1,height,width); obs=target.clone()
        if 'depth' in frame:
            with Image.open(io.BytesIO(payload[frame['depth']])) as im: raw=np.array(im)
            if frame.get('depth_encoding')=='float16_bits': raw=raw.astype(np.uint16).view(np.float16).astype(np.float32)
            else: raw=raw.astype(np.float32)
            raw*=float(frame.get('depth_scale',1.))
            raw[~np.isfinite(raw)|(raw<=0)]=0
            if 'depth_mask' in frame:
                with Image.open(io.BytesIO(payload[frame['depth_mask']])) as im: mask=np.asarray(im)>0
                if mask.shape!=raw.shape: raise ValueError('depth mask shape mismatch')
                raw[~mask]=0
            # Pixel partition before resizing prevents target pixels from entering inputs.
            salt=int(hashlib.sha256((row['group']+'/'+frame['depth']).encode()).hexdigest()[:16],16)
            held=np.random.default_rng(salt).random(raw.shape)<.2
            target=F.interpolate(torch.from_numpy(np.where(held,raw,0))[None,None],size=(height,width),mode='nearest')[0]
            obs=F.interpolate(torch.from_numpy(np.where(~held,raw,0))[None,None],size=(height,width),mode='nearest')[0]
        depths.append(target); inputs.append(obs)
    times=torch.tensor([f['time'] for f in chosen],dtype=torch.float64); times=times-times[0]
    cv=torch.tensor(camera_valid,dtype=torch.bool)[:,None]
    out=dict(rgb=torch.stack(images)[:,None],rgb_valid=torch.stack(image_valid)[:,None],times=times,camera_valid=cv,
             metric_valid=cv & bool(row.get('metric_valid',False)),time_valid=row.get('time_valid',True),
             timestamp_provenance=row['timestamp_provenance'],depth_provenance=row['depth_provenance'],
             scene_static=static_capture_evidence(row),static_scene_evidence=row.get('static_scene_evidence'),
             sequence=row.get('sequence'),archive_sha256=row.get('archive_sha256'),
             surface_correspondence_valid=False)
    if any(camera_valid): out.update(intrinsics=torch.stack(intrinsics)[:,None],camera_c2w=torch.stack(cameras)[:,None])
    if any('depth' in f for f in chosen):
        target=torch.stack(depths)[:,None]; obs=torch.stack(inputs)[:,None]
        out.update(depth=target,input_depth=obs,depth_valid=target>0,input_depth_valid=obs>0)
    return out
