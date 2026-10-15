# Data contract — MAVT chặng 1 (cổng G1)

**Phiên bản:** v1, 15/10/2026. **Phạm vi:** dữ liệu đang có trên DGX (`~/edtech/antoken`) và kim.long (lịch sử, không còn dùng).
Mỗi nguồn ghi rõ bằng chứng nào là nhãn thật, nhãn giả, hay không có. Thiếu nhãn là mask `False`, không bao giờ là giá trị 0. Thiếu hành động khác hành động bằng 0.

## Nguồn và bằng chứng

| Nguồn | Tầng · modality | Nhãn thật | Nhãn giả (ai tạo) | Không có → mask | Đơn vị · scale | Thời gian | Khoá split | Trên DGX |
|---|---|---|---|---|---|---|---|---|
| Open Images V7 train shard 1–2 | C · ảnh | – | – | camera, hình học, chuyển động | pixel, không metric | – | `img:<image id>` (tên file) | đang giải nén, manifest `openimages_v2.jsonl` |
| OpenVid-1M part 0 + 100 | B · video | – | caption, `aesthetic`, `motion`, `temporal_consistency`, `camera_motion` (tác giả OpenVid); track RAFT (dự kiến, Plan T14) | intrinsics, pose, depth | pixel, không metric | `fps`, `frames`, `duration` từ file; cửa sổ 16 frame, stride 2 | nguồn video: `yt:<id>`, `tt:<id>`, `<site>:<id>`; tên lạ → `clip:<stem>` | 35.117 clip, manifest `openvid_v2.jsonl` |
| Objaverse-LVIS triplane | asset 3D | – | caption Cap3D (31.669), còn lại tên lớp LVIS | depth, normal, view giữ riêng (GLB đã xoá) | chuẩn hoá khối đơn vị, **không cùng đơn vị với cảnh metric** | – | `obj:<uid>` | không có (chỉ ở kim.long) |
| nuScenes v1.0 (mini, keyframe trainval 01–10) | A · video đa camera + LiDAR | 6 camera, intrinsics, pose ego theo từng camera, LiDAR, box 3D có `instance_id` | – | sweep 12 Hz (chỉ có ở mini) | mét; gốc thế giới = khung ego tại frame tham chiếu (Plan §1.7) | timestamp µs; keyframe 2 Hz | scene chính thức train/val; điểm LiDAR giữ riêng theo `sha1(source/lidar_key)` (Plan T7) | có |
| OmniWorld-Game @ 4e34897 (7/479 cảnh, 8,4 GB) | B · video 1280×720, 24/30 fps, cảnh game | – | depth, pose camera (OpenCV world-to-camera), flow tiến, mask tiền cảnh và mask động, caption: nhãn do OmniWorld tạo, không phải log của engine; 183/479 cảnh không có flow | LiDAR, box, instance | đơn vị cảnh × `Metric Scale` trong metadata ≈ mét (depth trung vị đo được 7,5–20 m) | `split_info.json`: cảnh cắt thành split liên tục; `Test Split Index` của tác giả | theo cảnh (`uid`); không trộn split của một cảnh qua hai phía | có, `data/datasets/omniworld` |
| Imagenette2-320 val | chỉ đánh giá | nhãn lớp | – | – | pixel | – | tập ngoài | có |
| CIFAR-100 | chỉ đánh giá (kNN gate) | nhãn lớp | – | – | pixel 32² | – | tập ngoài | chưa kiểm |

## Quy tắc

1. **Split.** Ảnh, video, 3D dùng `split_mode: group_hash` (`src/mavt/data/splits.py`): mẫu thuộc phía nào do `sha1(salt + khoá nhóm)` quyết định. Salt `mavt-split-v1`. Tỉ lệ val tính theo **nhóm**: ảnh 0,005, video 0,02, 3D 0,01. Split không đổi khi manifest lớn thêm. Tăng tỉ lệ chỉ chuyển mẫu từ train sang val. Kiểm bằng `scripts/data/check_split.py --split-mode group_hash`: mục tiêu là 0 nhóm nằm ở cả hai phía. nuScenes dùng split scene chính thức; không trộn view, render hay frame của cùng scene/asset qua hai phía.
2. **Đánh giá.** `eval_gates.py` và `eval_modalities.py` chỉ lấy mẫu từ tập val của split trên, cùng salt và tỉ lệ với cấu hình train. Số đo trước 15/10 lấy từ N dòng đầu/cuối manifest, tức chủ yếu là mẫu train (E1.2).
3. **Manifest có phiên bản.** Mỗi lần dựng: `*_v2.jsonl` kèm số dòng, sha256 và file `*.bad.jsonl` các mẫu bị loại (file hỏng, video dưới 16 frame). Đổi nội dung thì tăng hậu tố phiên bản, không ghi đè lặng lẽ.
4. **Cache.** Khoá cache = (nguồn, phiên bản manifest, phiên bản tiền xử lý, teacher + revision, `sample_id`, timestamp, tập quan sát dùng làm input). Teacher được đọc frame tương lai **chỉ để tạo đích**; feature tương lai không bao giờ vào input của prefix hay vào dựng graph.
5. **Augmentation hình học** phải cập nhật camera, tia, depth và tương ứng đi kèm. Không crop/flip từng frame độc lập khi mẫu có camera hoặc track.

## Còn thiếu so với roadmap

- Loader v3 trả `{data, modality, caption, id}`, chưa tách `observations / targets / metadata / validity_masks`. Với nuScenes, phần này là `SceneSample` của Plan 4D (T4).
- Kiểm pairing, camera và timestamp của nuScenes (E1.1) chưa chạy.
- Bộ test cố định (E1.3) chưa dựng.
- Sai số nhãn hình học giả so với LiDAR (E1.4) chưa đo.
- Khi một file hỏng lúc decode, `ManifestImageDataset` / `ManifestVideoDataset` thay bằng một mẫu ngẫu nhiên của toàn manifest, có thể thuộc phía split bên kia. Manifest đã được scan sạch nên trường hợp này hiếm; sửa khi loader chọn mẫu thay trong cùng split.
- Ảnh/asset trùng nội dung nhưng khác id chưa được gộp nhóm; `check_split.py` đo tần suất trường hợp này.
