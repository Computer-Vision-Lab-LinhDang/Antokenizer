# Redesign 2 trục: bất biến / biến động — kiểm chứng trước khi xây

*Đo trên video thật của dự án (`download_sample/before/*.mp4`, 2 clip, 16 frame, 256²) và ảnh GT thật. Script: `audit6.py`, `audit7.py`.*

---

## 0. Kết luận ngắn

Đề xuất của bạn **đúng ở chỗ quan trọng nhất và sai ở một chỗ có thể sửa được**.

| Thành phần đề xuất | Kiểm chứng | Kết luận |
|---|---|---|
| Giảm 4D → 2 loại quan hệ | cắt **67%** overhead RGAT, giải phóng 4.7 M tham số | **đúng** |
| Trục "bất biến" tồn tại thật | chiếm **97.6–98.8%** năng lượng clip | **đúng, mạnh hơn tôi tưởng** |
| Phân tách hai trục nén thật | tiết kiệm **33–56%** rank so với mã hoá từng frame | **đúng** |
| Entropy làm tín hiệu cấp ngân sách | Gini **0.62–0.76**, ổn định giữa hai nửa clip (Spearman **0.93–0.95**) | **đúng, dùng được ngay** |
| "residual" và "temporal" là hai **trục song song** | hai trục **bổ trợ**, không thay thế nhau | **cần sửa lại phát biểu** |
| Một trục "biến động" dùng chung cho cả 3 modality | ảnh tĩnh **không có** trục nào có thống kê tương đương | **đây là lỗ hổng chính** |

Điểm quan trọng nhất, và nó ngược trực giác: **biến động thời gian không phải tần số cao.** Đo được 54% năng lượng của nó ở **tần số thấp**, chỉ 5% ở tần số cao. Trong khi residual hạng thấp (trục cũ) thì ngược lại: 37% cao tần, 1% thấp tần. Nghĩa là hai trục này **không đo cùng một thứ** — và đó thực ra là tin tốt, xem mục 2.

![Kiểm chứng đề xuất 2 trục trên dữ liệu video thật]({{artifact:art_959079dc-752a-413b-8fd5-9774493ad764}})

---

## 1. Cái đề xuất giải đúng: cơ chế sinh thuế

Thuế "unified" trong thiết kế cũ đến từ chỗ này: **một** latent z phải đồng thời (a) bất biến để mang ngữ nghĩa, và (b) chi tiết để tái tạo pixel. Hai yêu cầu đối nghịch trên **cùng một tập số**, nên gradient của hai loss kéo ngược chiều nhau — và số của AToken cho thấy recon thắng dần, semantic thoái dần đều qua ba stage.

Đề xuất của bạn giải đúng chỗ đó: **tách thành hai nhánh có invariance khác nhau về bản chất**, rồi cho mỗi loss tác động lên nhánh phù hợp. Semantic loss lên nhánh bất biến; recon loss lên cả hai. Đây là điểm khác biệt thật so với thiết kế cũ — C-D split cũ *cũng* nói tách theo invariance, nhưng cả hai nhánh đều xuất phát từ **cùng một** phép chiếu không gian, nên "bất biến" chỉ là mong đợi, không phải tính chất cấu trúc. Ở đề xuất mới, tính bất biến của nhánh 1 là **định nghĩa** (trung bình dọc trục biến động), không phải hy vọng.

Số ủng hộ mạnh:

| clip | năng lượng bất biến | năng lượng biến động | % patch tĩnh hoàn toàn |
|---|---:|---:|---:|
| 00013001 | **97.6%** | 2.4% | 43.4% |
| 00013006 | **98.8%** | 1.2% | 37.5% |

Và phân tách này **nén thật**, không chỉ đổi tên. Đo bằng rank cần để giữ 99% năng lượng:

| clip | mã hoá 16 frame độc lập | hai trục (bất biến + lệch) | tiết kiệm |
|---|---:|---:|---:|
| 00013001 | 496 | 216 | **56%** |
| 00013006 | 323 | 218 | **33%** |

Đây là con số mà thiết kế cũ **chưa từng có** — C-D split cũ ở default không nén gì cả (320 token cho 256 patch).

---

## 2. Chỗ cần sửa lại phát biểu: hai trục không thay thế nhau

Bạn phát biểu "residual học phần dư bất biến, temporal học biến động". Nhưng đo ra thì:

| | tần số **thấp** | tần số **cao** |
|---|---:|---:|
| trục CŨ (residual hạng thấp) | 1% | **37%** |
| trục MỚI (lệch so với time-mean) | **54%** | 5% |

**Biến động thời gian chủ yếu là cấu trúc thấp tần *di chuyển*** — một vật thể dịch chuyển tạo ra hiệu số lớn ở vùng biên nhưng bản thân hiệu số đó trơn, không phải texture. Còn residual hạng thấp bắt đúng texture tĩnh.

Hệ quả thiết kế: nếu bạn **thay** residual bằng temporal, bạn **mất** kênh chi tiết tần số cao — chính là thứ quyết định rFID/LPIPS. Nhưng nếu bạn giữ **cả hai** thì đúng: hai trục phân hoạch tín hiệu theo hai chiều gần như trực giao.

Nên phát biểu đúng của kiến trúc là **ba nhánh, không phải hai**:

```
z_inv    = bất biến dọc trục biến động        → ngữ nghĩa   (semantic loss ở đây)
z_var    = biến động dọc trục đó              → chuyển động, thấp tần
z_detail = residual trong-mẫu, cao tần        → texture     (recon loss nặng ở đây)
```

Đây vẫn là **2 loại quan hệ** cho RGAT (bạn đúng ở phần đó — chỉ cần "cùng-vị-trí-khác-thời-điểm" và "khác-vị-trí-cùng-thời-điểm"), nhưng **3 kênh latent**. Hai chuyện độc lập nhau: số loại cạnh trong graph ≠ số nhánh latent.

---

## 3. Lỗ hổng chính: ảnh tĩnh không có trục biến động

Đây là chỗ đề xuất chưa đứng được, và tôi nghĩ cần giải trước khi xây.

"Handle được cả 3 modality" đòi hỏi trục "biến động" có nghĩa cho cả ba. Video có trục thời gian tự nhiên. **Ảnh tĩnh thì không có gì.** Tôi thử ba trục thay thế và đo thống kê so với trục thời gian:

| trục ứng viên | K | % năng lượng ở nhánh biến động | Gini | so với video |
|---|---:|---:|---:|---|
| **video: thời gian** | 16 | **2.4%** | 0.76 | (mốc) |
| ảnh: kim tự tháp scale | 4 | 1.4% | 0.59 | gần nhất |
| ảnh: dải tần số | 3 | 66.7% | 0.37 | **lệch 28×** |
| ảnh: láng giềng không gian | 4 | 43.9% | 0.35 | **lệch 18×** |

Trục tần số và trục láng giềng cho phân bố hoàn toàn khác — 44–67% năng lượng ở nhánh "biến động" so với 2.4% của video. Một nhánh dùng chung sẽ thấy hai phân bố lệch nhau hàng chục lần giữa các modality, tức lại tái tạo **đúng vấn đề** mà `ModalityEMAWeighter` đang cố vá (và vá sai chiều — audit trước, H7).

Chỉ **kim tự tháp scale** có thống kê gần video (1.4% vs 2.4%, Gini 0.59 vs 0.76). Đó là ứng viên khả dĩ nhất: "biến động dọc trục độ phân giải" — ảnh mất gì khi giảm scale. Nó cũng có nghĩa vật lý sạch và nối được với multi-resolution.

**Ba cách xử lý, theo mức rủi ro:**

1. **Kim tự tháp scale làm trục biến động cho ảnh** (rủi ro thấp nhất). Ảnh: K=4 mức scale. Video: K=T frame. 3D: K=3 plane. Cùng một cơ chế, cùng thống kê xấp xỉ. Cần đo thêm trên nhiều ảnh trước khi cam kết.
2. **Ảnh = trường hợp suy biến K=1** (rủi ro trung bình). Nhánh biến động rỗng cho ảnh, `z_var` là zero, chỉ `z_inv` + `z_detail` hoạt động. Trung thực về mặt toán học, nhưng nghĩa là ảnh chạy một kiến trúc khác — mất đi tính "unified" mà đề xuất muốn có.
3. **Trục biến động = augmentation** (rủi ro cao). Hai view augment của cùng ảnh là "biến động" nhân tạo. Nhưng repo **không có augmentation nào** (chỉ Resize/CenterCrop/Normalize), nên đây là công việc phải xây thêm, và trục sẽ mang thống kê của augmentation ta chọn — tức ta tự định nghĩa cái mình muốn đo.

Tôi nghiêng phương án 1, với điều kiện đo thêm trên ≥100 ảnh (2 clip + 1 ảnh hiện tại là quá mỏng để cam kết kiến trúc).

---

## 4. Entropy làm tín hiệu cấp ngân sách: dùng được ngay

Đây là phần đề xuất của bạn mà số liệu ủng hộ **rõ nhất**, và tôi nghĩ nó đáng giá hơn cả phần RGAT.

Biến động **rất tập trung**:

| clip | entropy / max entropy | Gini | trùng top-25% giữa 2 nửa clip | Spearman |
|---|---:|---:|---:|---:|
| 00013001 | 0.792 | 0.761 | 86% | **0.945** |
| 00013006 | 0.876 | 0.619 | 73% | **0.927** |

Spearman 0.93–0.95 giữa hai nửa clip nghĩa là: **bản đồ biến động ổn định**, đo ở nửa đầu vẫn dự đoán tốt nửa sau. Đây là điều kiện cần để cấp ngân sách theo entropy thay vì theo tỉ lệ cố định.

Và tỉ lệ cố định hiện tại (`0.25·N = 64`) **sai cả hai chiều**:

| clip | patch cần cho 50% biến động | cho 90% | cho 99% | tỉ lệ cố định cấp |
|---|---:|---:|---:|---:|
| 00013001 | 19 | 78 | 153 | 64 |
| 00013006 | 37 | 114 | 183 | 64 |

Với clip 1, 64 token là **quá nhiều** cho 50% biến động (chỉ cần 19) nhưng **không đủ** cho 90% (cần 78). Một hằng số không thể đúng cho cả hai. Ngân sách thích ứng theo entropy là cải tiến đo được, không cần chờ kiến trúc mới.

**Cảnh báo cần thiết:** ngân sách động làm số token thay đổi theo mẫu. Điều đó (a) phá vỡ cache adjacency `(modality, N)` — vốn đã sai, giờ sai mỗi step (H6 và audit4 mục C: hai mask khác nhau cùng N cho 9.31% cạnh sai); (b) làm batch không đồng nhất chiều, cần padding + mask; và (c) `StandardTransformerBlock` gọi `F.scaled_dot_product_attention` **không có** tham số `attn_mask`, nên phải sửa mọi block. Đây là công việc thật, không phải chi tiết.

---

## 5. RGAT 2 loại cạnh: rẻ hơn 67%

| | tham số |
|---|---:|
| block transformer thường | 7 087 872 |
| RGAT 4 loại cạnh (hiện tại), overhead | **3 535 936** |
| RGAT 2 loại cạnh (đề xuất), overhead | **1 176 608** |
| **giải phóng** (2 block RGAT) | **4 718 656** |

Bạn đúng: bỏ `depth` (zeros tensor cho **cả ba** modality — đã đo) và `cross_plane` (64% cạnh là giả do zero-padding) cắt 67% overhead mà không mất thông tin nào đang thực sự được dùng.

Nhưng lưu ý ở mục 2: điều này vẫn **không** làm RGAT thắng relative-position-bias về chi phí — 1.18 M so với 6 000 vẫn là 196×. Câu hỏi "RGAT có xứng chi phí" vẫn cần ablation, chỉ là câu hỏi đã nhỏ hơn 3 lần.

Về cách RGAT đọc data: nếu đi hướng này thì **encoding cũng nên đổi**. Toạ độ 4D `(t,x,y,z)` với zero-padding là nguồn của 64% cạnh giả. Thay bằng `(vị_trí_trong_mẫu, chỉ_số_dọc_trục_biến_động)` — 2 chiều, không có chiều nào phải zero-pad, và cùng một encoding cho cả ba modality (ảnh: `(patch, mức_scale)`; video: `(patch, frame)`; 3D: `(patch, plane)`). Đây là phần "tái cấu trúc cách nó đọc data" mà tôi nghĩ là điểm mạnh nhất của đề xuất — nó loại bỏ zero-padding ở gốc chứ không vá hệ quả.

---

## 6. Đề xuất kiến trúc cụ thể

```
INPUT  →  patchify  →  positions = (p, k)
          p = chỉ số patch trong mẫu
          k = chỉ số dọc trục biến động
              ảnh:  mức kim tự tháp scale (K=4)
              video: frame            (K=T)
              3D:    plane            (K=3)

BACKBONE  12 block, 2 block RGAT với ĐÚNG 2 loại cạnh:
          - cùng p, khác k   ("biến động")
          - khác p, cùng k   ("không gian")

SPLIT     z_inv    = pool_k(feat)              # bất biến, ~98% năng lượng
          z_var    = feat − broadcast(z_inv)   # biến động, thấp tần
          z_detail = feat − rank_r(feat)       # residual cao tần trong-mẫu

BUDGET    n_var  = f(entropy của z_var)        # thích ứng, không cố định
          n_det  = f(entropy của z_detail)

LOSS      semantic  → CHỈ z_inv
          recon     → cả ba
          KL        → cả ba, nhưng riêng weight
```

**Vì sao cấu trúc này giảm thuế:** semantic loss không còn tác động lên kênh nào phải mang chi tiết pixel. Trong thiết kế cũ, gradient của distillation và của L1 gặp nhau trên cùng 10 240 số. Ở đây chúng tách về hai tập tham số khác nhau, chỉ gặp nhau qua backbone chung — chỗ mà chia sẻ biểu diễn là điều ta *muốn*.

**Vì sao nó vẫn có thể thất bại** (ba rủi ro tôi thấy):

1. `z_inv` với K=4 (ảnh) thì "bất biến" yếu hơn nhiều so với K=16 (video). Ngữ nghĩa của ảnh có thể vẫn chưa đủ tín hiệu.
2. Ba nhánh nghĩa là ba chỗ có thể collapse. Cơ chế chống collapse duy nhất trong repo (`slot_diversity_loss`) hiện là **no-op** (H8) — phải sửa trước.
3. Ngân sách động + attention mask là thay đổi hạ tầng lớn. Nếu làm nửa vời (ngân sách động nhưng cache adjacency chưa sửa) thì kết quả sẽ tệ hơn baseline mà không rõ vì sao.

---

## 7. Thứ tự tôi khuyên

| # | Việc | Vì sao trước |
|---|---|---|
| 1 | Đo trục kim tự tháp scale trên ≥100 ảnh thật | quyết định mục 3 — nếu trục này không khớp thống kê thì "unified" phải định nghĩa lại |
| 2 | Sửa cache adjacency → key theo hash(positions) | mọi thứ sau đều cần |
| 3 | Bật ngân sách theo entropy trên **video** trước (trục có sẵn) | rẻ nhất, đo được ngay, không cần kiến trúc mới |
| 4 | Sửa `slot_diversity_loss` khỏi `no_grad` | ba nhánh = ba chỗ collapse |
| 5 | Prototype 3 nhánh trên video ở scale nhỏ | video là modality có trục thật, dễ debug nhất |
| 6 | Mở rộng sang ảnh + 3D sau khi video chạy | tránh debug 3 modality cùng lúc |

**Điều tôi vẫn giữ nguyên từ lần trước:** 19 576 tham số/mẫu (195.8 M trên 10k ảnh) nghĩa là bàn thí nghiệm chưa phân xử được giữa các lựa chọn kiến trúc. Đề xuất này hay hơn thiết kế cũ về mặt lập luận và có số ủng hộ, nhưng để *chứng minh* nó hơn thì cần dữ liệu nhiều hơn và bộ đo (rFID + linear probe) mà repo hiện chưa có.

---

## 8. Giới hạn của kiểm chứng này

Cần nói rõ để không dùng quá số liệu:

- **2 clip video + 1 ảnh.** Đủ để loại bỏ giả thuyết sai (trục tần số/láng giềng lệch 18–28× là kết luận khó đảo), không đủ để cam kết hằng số kiến trúc.
- **Patch pixel thô, không phải feature đã train.** Không có checkpoint backbone ở môi trường này. Feature đã train sẽ có phổ khác — chiều kết luận nên vững, độ lớn thì không.
- **3D chưa đo được.** Repo không có render triplane thật nào; phần 3D trong đề xuất là suy luận theo cấu trúc (K=3), chưa có số.
- **"Entropy" ở đây là entropy của phân bố năng lượng biến động theo patch**, không phải entropy mã hoá (bits). Nếu bạn muốn nghĩa thứ hai (rate-distortion thật) thì cần một entropy model, và đó là hạng mục riêng.

---

*Script: `audit6.py` → `audit6.json`; `audit7.py` → `audit7.json`. Kết hợp với `MAVT_DESIGN_CRITIQUE.md` (phê bình thiết kế cũ) và `MAVT_ARCH_HOLES.md` (18 lỗ hổng thực thi).*
