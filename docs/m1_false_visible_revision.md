# Sửa đổi thước đo false-visible của cổng M1 (E1.1)

**Trạng thái:** đề xuất, ngày 15/10/2026. Người dùng cần duyệt trước khi chép vào spec. Spec gồm §7.1 của
`research/problem_formulation.md` và mục T27 của `implementation_plan.md`; cả hai file không nằm trong repo này.

**Lý do sửa:** trên cảnh tổng hợp có visibility chính xác, luật cũ đánh dấu false-visible cho 59% số điểm thật
ra nhìn thấy được, khi gộp ±5 sweep. Như vậy cạnh đúng hoàn toàn vẫn trượt cổng. Sai số này đến từ mặt đất nhìn
xiên. Bằng chứng ở bên dưới.

Không số cũ nào bị thay. Báo cáo E1.1 ghi cả hai luật cạnh nhau.

## Luật cũ (T27, như đo lần đầu)

Một điểm thành viên nằm sau cạnh PROJ_P2H đã giữ được coi là false-visible khi có điểm tham chiếu nằm cùng ô
4 px với nó và gần camera hơn quá 0,5 m. Tham chiếu là LiDAR gộp ±5 sweep. Đây là đúng luật z-buffer mà bộ dựng
cạnh dùng, chỉ áp lên mây dày hơn.

## Luật đề xuất (`c4+rel+plane`)

Vẫn ô 4 px, nhưng có hai thay đổi:

- **Dung sai theo độ sâu:** dung sai là 0,5 m + 5% độ sâu của điểm.
- **Bỏ điểm cùng mặt:** không xét điểm tham chiếu nằm trong 0,2 m quanh mặt phẳng hub của điểm. Mặt phẳng hub
  là mặt PCA mà bộ dựng cạnh đã tính, nên điểm cùng mặt không bị coi là vật che.

Code nằm ở `mavt4d.eval.graph_checks.FVRule(cell_px=4, tol_m=0.5, tol_rel=0.05, plane_tau=0.2)` và
`rule_visible`. Cảnh tổng hợp và nuScenes-mini chạy cùng một code này.

## Bằng chứng trên cảnh tổng hợp có visibility chính xác

Nguồn số: `results/e1_1/fv_calibration.md`, commit `0d8d637`.

**Cảnh tổng hợp:**

- Hình học giống m3 của nuScenes: 6 camera 448×256, fx ≈ 360 px, pitch 0.
- LiDAR 32 tia, các sweep cách nhau 50 ms, render từ cùng một cảnh.
- Ego chạy 8 m/s; cảnh có 8 hộp, 1 hộp chuyển động.
- Chạy cùng pipeline với mini: hold-out 0,3, rồi hub, rồi graph mặc định.
- Sweep lân cận giữ hộp đứng yên, vì chúng đóng vai tường và cột không có nhãn vẫn còn trong mây ±5 của nuScenes.
  Chỉ bỏ điểm thuộc hộp chuyển động.

**Hai tập seed:**

- Luật được chọn trên 16 seed hiệu chỉnh (0–15).
- Mọi số dưới đây lấy từ 16 seed báo cáo (16–31), không dùng cho lựa chọn nào.

**Các tập điểm thành viên:**

- **GT-đúng:** điểm thật sự nhìn thấy theo `gt_visible`. Tỉ lệ false-visible thật của tập này bằng 0, nên mọi
  điểm luật đánh dấu đều là lỗi của luật.
- **GT-sai:** điểm thật sự bị che. Đây là các cạnh sai đã biết mà luật phải bắt được.
- **Cạnh của bộ dựng:** điểm thành viên sau các cạnh mà bộ dựng giữ lại. Tỉ lệ false-visible thật của tập này
  là 1,27%, tức 1.630 trên 128.428 điểm.

Bảng dưới đây so tỉ lệ mỗi luật báo ra với độ dày tham chiếu: chỉ sweep keyframe, gộp ±1 và gộp ±5 sweep.

| Hàng | Luật | Tập điểm | Keyframe | ±1 | ±5 | Recall trên GT-sai (keyframe / ±1 / ±5) |
|---|---|---|---|---|---|---|
| (1) | cũ | GT-đúng | 0,29% | 1,26% | **59,18%** | 32,15% / 42,47% / 88,53% |
| (2) | cũ | cạnh của bộ dựng (thật 1,27%) | 0,21% | 1,36% | **59,49%** | |
| (3) | đề xuất | GT-đúng | 0,22% | 0,36% | **0,69%** | 31,87% / 41,96% / 82,17% |
| (4) | đề xuất | cạnh của bộ dựng (thật 1,27%) | 0,18% | 0,49% | **1,49%** | |
| (5) | đề xuất | nuScenes-mini, cạnh của bộ dựng | 2,07% | 7,21% | **15,83%** | không có GT |
| (5') | cũ | nuScenes-mini, cạnh của bộ dựng | 4,28% | 15,60% | **30,66%** | không có GT |

**Đọc bảng:**

- **Luật cũ sai vì mặt đất xiên.** Ở ±5, lỗi tập trung ở mặt nhìn xiên: 74% ở |cos| < 0,1 và 30% ở 0,1–0,3, còn
  ở |cos| ≥ 0,3 chỉ khoảng 1%. Lỗi cũng tăng theo độ sâu: 1,2% dưới 10 m và 81% trên 40 m. Gộp càng nhiều sweep
  thì mặt đất càng có thêm điểm gần hơn trong cùng ô, nên lỗi tăng theo độ dày tham chiếu.
- **Recall 88,5% của luật cũ ở ±5 không có nhiều thông tin.** Luật này đánh dấu 59% số điểm đúng, nên recall cao
  một phần chỉ vì nó đánh dấu gần như mọi thứ.
- **Luật đề xuất vẫn giữ được lợi ích của tham chiếu dày.** Recall của nó tăng theo độ dày (31,9% → 42,0% →
  82,2%), còn lỗi gần như không đổi (0,22% → 0,36% → 0,69%).
- **Chỗ luật đề xuất còn sai: ô nằm trên biên vật thể.** Ở những ô có nhiều mặt GT, lỗi còn 32,5%. Các ô này chỉ
  chiếm 2.157 trên 130.100 điểm GT-đúng. Ô một mặt chỉ còn 0,15%.
- **Luật đề xuất bỏ sót vật che rất sát.** Điểm GT-sai nằm sau mặt GT dưới 1 m bị bỏ sót hết, nhưng chỉ có 6 điểm
  như vậy. Với điểm sau 1–5 m, recall là 37,7%; sau hơn 5 m là 84,1%.

**Bảng lưới tám luật** (ô 4 hoặc 2 px; có hoặc không dung sai tương đối; có hoặc không lọc mặt phẳng) nằm trong
`fv_calibration.md`.

- Tiêu chí chọn là Youden J = recall − lỗi ở ±5 trên seed hiệu chỉnh. Luật `c4+rel+plane` đạt J = +0,638. Luật cũ
  đạt +0,306.
- **Ghi nhận thay đổi tiêu chí.** Tiêu chí đặt trước khi chạy là recall không kém luật cũ quá 5 điểm. Tiêu chí đó
  suy biến, vì recall của luật cũ bị thổi phồng do đánh dấu quá nhiều. Tiêu chí đã đổi sau khi xem seed hiệu chỉnh
  1, trước khi chạy bất kỳ seed báo cáo nào. Cùng lúc đó còn hai thay đổi: giữ hộp đứng yên trong sweep lân cận,
  và tăng từ 4 + 4 lên 16 + 16 seed.

## Điều chưa giải quyết

**Trên nuScenes-mini, luật đề xuất vẫn KHÔNG ĐẠT cổng.** False-visible là 15,83%, so với ngưỡng < 5%. Tỉ lệ vẫn
tăng theo độ dày tham chiếu (2,07% → 7,21% → 15,83%), trong khi trên cảnh tổng hợp thì gần như phẳng. Phần còn lại
là một trong hai thứ:

- **False-visible thật của bộ dựng trên dữ liệu thật**; hoặc
- **Hiệu ứng mà cảnh tổng hợp không có:**
  - sweep lệch pha nhau, méo do chuyển động trong một sweep và rolling shutter;
  - bề mặt không phẳng như cây cối, khiến hub PCA không đại diện;
  - vật chuyển động không có nhãn.

Lần đo này không tách được hai khả năng trên. Muốn tách cần GT visibility trên dữ liệu thật, hoặc một cảnh tổng
hợp có thêm các hiệu ứng đó.

**Các biến thể bộ dựng đều không đạt.** Đo bằng luật đề xuất ở ±5:

| Biến thể | False-visible |
|---|---|
| Lọc mặt xiên | 14,98% |
| Dung sai 5% độ sâu | 15,84% |
| Ô 2 px | 19,82% |
| Lọc mặt xiên + dung sai 5% | 14,96% |
| Cả ba | 18,83% |

**Lưu ý về pháp tuyến hub:** 1,41% số cạnh được giữ đến từ hub gần thẳng hàng (λ_mid / λ_max < 0,05). Với các hub
này, phép thử mặt phẳng của luật đề xuất là tuỳ ý.

## Cách tái lập

```bash
python scripts/mavt4d/calibrate_false_visible.py --out results/e1_1
```

```bash
PYTHONPATH=src python scripts/mavt4d/check_nuscenes.py --dataroot $NUSC --version v1.0-mini --out results/e1_1
```

Lệnh đầu chạy được trên CPU, khoảng 1 phút. Lệnh thứ hai chạy trên headnode DGX, khoảng 11 phút với 8 worker.
