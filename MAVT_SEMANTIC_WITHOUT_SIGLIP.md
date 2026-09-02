# Lấy semantic mà không dựa vào SigLIP — phương án cho MAVT

*Audit trên code thật `atoken/src/mavt/` (HEAD `87dbdcf`). Ràng buộc và con số đo bằng `audit4.py`.*

---

## 0. Trước hết: SigLIP đang làm gì trong MAVT, và bỏ nó mất gì

Hiện tại có **hai** đường SigLIP2, dễ bị lẫn:

| Đường | Vai trò | Nếu bỏ |
|---|---|---|
| `backbone.load_siglip2_weights()` | **khởi tạo** 10 transformer block | mất prior, phải train backbone từ đầu |
| `semantic_teacher` + `cosine_distill_loss` | **supervision** cho understanding head | mất toàn bộ tín hiệu semantic |

Câu hỏi của bạn nhắm vào đường thứ hai. Nhưng cần nói rõ: **bỏ đường 2 mà giữ đường 1 thì vẫn là dựa vào SigLIP** — chỉ là dựa vào lúc init thay vì lúc train. Nếu mục tiêu là độc lập hoàn toàn thì cả hai phải thay, và đó là bài toán lớn hơn nhiều (train backbone from scratch cần dữ liệu cỡ ImageNet-1k trở lên; hiện có 10k ảnh).

Một điểm quan trọng từ audit trước: **distillation hiện tại yếu hơn con số báo cáo.** `cos_sim teacher = 0.815` trong `results.md` được đo trên đường có VAE sampling, và hai lần forward cùng một ảnh cho semantic lệch nhau cosine **0.705**. Nghĩa là phần lớn "khoảng cách tới 1.0" không phải do distillation chưa hội tụ mà do nhiễu sampling. Sửa cái đó (dùng μ ở eval) rẻ hơn mọi phương án dưới đây và nên làm trước khi kết luận SigLIP distillation không hiệu quả.

---

## 1. Ràng buộc thật quyết định phương án nào khả thi

Đây là phần hay bị bỏ qua. Không phải ý tưởng nào hay hơn — mà ý tưởng nào **chạy được với code và data hiện có**.

### Ràng buộc 1: batch hiệu dụng = 16

`stage2_universal.yaml`: `batch_size: 8` × `accumulate_grad_batches: 2`.

**`accumulate_grad_batches` không thêm negative.** InfoNCE cần negative trong *cùng một forward*; gradient accumulation chỉ cộng gradient của các forward riêng biệt.

| | số mẫu trong 1 forward | chance top-1 |
|---|---:|---:|
| MAVT hiện tại | 16 | 6.25% |
| ngưỡng InfoNCE dùng được | ~1 024 | 0.1% |
| SigLIP (paper) | 16 384 | 0.006% |
| CLIP (paper) | 32 768 | 0.003% |

Đo thực: InfoNCE với cặp đã align một phần cho loss 2.41 ở B=16 so với random 2.96 — khoảng phân giải chỉ 0.55 nat. Ở B=1024, khoảng đó là 6.45 vs 7.07. Batch 16 gần như không phân biệt được "đã học" với "chưa học".

**Hệ quả: mọi phương án contrastive (image-text, SimCLR) bị chặn cứng** — không phải vì ý tưởng sai mà vì hạ tầng. Đây là ràng buộc đầu tiên phải giải nếu muốn đi đường contrastive, và nó vướng H12 (RGAT dense O(N²) tốn 6.44 GB cho video B=8) — tức phải sửa RGAT trước khi tăng batch.

### Ràng buộc 2: không có augmentation nào

`datasets.py` chỉ có `Resize → CenterCrop → ToTensor → Normalize`. **Không một random transform nào.**

Mọi phương án joint-embedding (BYOL, DINO, SimCLR, VICReg, Barlow Twins) đều cần hai view khác nhau của cùng một mẫu. Không augmentation → không có view thứ hai → các phương án đó không chạy được cho ảnh.

### Ràng buộc 3: caption có sẵn nhưng chưa được nối

| | trạng thái |
|---|---|
| caption cho image / video / 3D | **có** (`captions/*.json`, và `.txt` trong WDS shard) |
| `infonce_loss` | **đã implement** trong `losses.py` |
| `text_embed` trong `MAVTLoss.forward` | đã có tham số |
| `_step` đọc `batch['caption']` | **không** |
| text encoder trong repo | **không** |

Nghĩa là: dù bật `use_clip=True`, `text_embed` vẫn là `None` → `l_clip = 0` vĩnh viễn. Text path đứt ở hai chỗ (không đọc caption, không có encoder), không chỉ ở cờ config.

### Ràng buộc 4: positive pair miễn phí có sẵn — nhưng không cho ảnh

| modality | view thứ hai miễn phí | số cặp/mẫu |
|---|---|---:|
| video | frame khác trong cùng clip | **120** (16 frame) |
| 3D triplane | plane khác của cùng object | **3** |
| image | — | **0** |

Đây là tài sản chưa dùng. Hai frame của một clip là hai view của cùng một nội dung ngữ nghĩa — chính xác là positive pair mà SimCLR phải tạo bằng augmentation.

---

## 2. Bốn phương án, xếp theo mức khớp với ràng buộc

### Phương án A — Masked latent prediction (JEPA-style) ⭐ khuyến nghị chính

**Ý tưởng:** che một phần token, buộc model dự đoán **biểu diễn** (không phải pixel) của phần bị che, từ phần còn thấy. Semantic nổi lên vì muốn đoán được vùng bị che thì phải hiểu nội dung, không thể nội suy tần số cao.

**Vì sao khớp với MAVT:**
- **Không cần batch lớn.** Signal per-sample, không cần negative.
- **Không cần augmentation.** Mask chính là view thứ hai.
- **Token dropping đã native.** `patchify` trả `positions` **riêng** khỏi tokens, nên lấy subset token + subset positions là chạy được ngay — không cần plumbing attention mask (điều này quan trọng vì `StandardTransformerBlock` gọi `F.scaled_dot_product_attention(Q,K,V, dropout_p=0.0)` **không có** tham số `attn_mask`, thêm mask vào sẽ phải sửa mọi block).
- **Có sẵn chỗ gắn.** `latent_token_types` phân biệt content/detail: đặt objective ngữ nghĩa lên content slot, objective cục bộ lên detail token.
- Dùng chung ngôn ngữ với thiết kế hiện tại: MAVT vốn đã là kiến trúc encoder-latent-predictor.

**Chặn cứng phải sửa trước:** cache adjacency `(modality, N)`. Hai mask ngẫu nhiên **khác nhau** nhưng cùng số token còn lại sẽ **dùng chung** một mask hình học:

| | giá trị |
|---|---:|
| hai mask khác nhau, cùng N=128, adjacency có giống nhau? | **không** |
| tỉ lệ cạnh sai nếu dùng cache | **9.31%** |

Đây là cùng bug H6 trong audit trước, nhưng masking làm nó trở thành lỗi *mỗi step* thay vì lỗi thỉnh thoảng. Phải đổi key thành hash của tập positions trước khi bật masking. Ngoài ra ở keep_frac=0.25, đã xuất hiện token bị cô lập (degree 0) — RGAT có `nan_to_num` xử lý, nhưng nên kiểm tra tỉ lệ này khi chọn mask ratio.

**Chi phí:** thêm một predictor nhỏ (transformer 2–4 block) + logic mask. Không cần model ngoài.

---

### Phương án B — Positive pair từ chính dữ liệu (temporal & cross-plane) ⭐ khuyến nghị làm song song

**Ý tưởng:** hai frame của cùng clip phải có content slot giống nhau; ba plane của cùng object cũng vậy. Ép cosine giữa content slot của hai view, không cần negative (chỉ cần thêm một cơ chế chống collapse — xem dưới).

**Vì sao khớp:** miễn phí 120 cặp/clip, không cần augmentation, không cần model ngoài, code thêm rất ít.

**Điểm hay riêng cho MAVT:** đây là supervision **duy nhất** trong các phương án cho semantic *đúng bản chất modality*. Hiện tại `_make_teacher_input` lấy frame giữa cho video và plane XY cho 3D — nghĩa là "semantic của video" thực chất là semantic của một ảnh 2D chọn tuỳ ý. Cross-frame consistency cho semantic *của cả clip*; cross-plane cho semantic *của cả object*.

**Cảnh báo:**
- Chỉ áp được cho video và 3D. Ảnh không có view thứ hai → cần kết hợp với A.
- 52% video shard hỏng (theo `STAGE2_ANALYSIS.md`) → cặp positive từ tensor zeros là positive giả. **Phải lọc dữ liệu trước.**
- Cần chống collapse: chỉ ép giống nhau thì nghiệm tầm thường là mọi thứ ra cùng một vector. Đây là chỗ VICReg/Barlow Twins (variance-covariance) dùng được — không cần negative, chỉ cần penalize phương sai thấp và tương quan giữa chiều. Rẻ hơn InfoNCE và không đòi batch lớn.
- Lưu ý: `slot_diversity_loss` — cơ chế chống collapse hiện có — đang là **no-op** vì tính trong `torch.no_grad()` (H8). Phải sửa cái đó trước, nếu không phương án B sẽ collapse mà không ai thấy.

---

### Phương án C — EMA self-distillation (BYOL / DINO trên encoder của chính mình)

**Ý tưởng:** giữ một bản copy EMA của encoder làm teacher, thay cho SigLIP2 đóng băng. Teacher tiến hoá cùng student → không phụ thuộc model ngoài.

**Chi phí bộ nhớ:** encoder MAVT là **93.9 M** tham số, bản EMA fp32 tốn **0.376 GB** — xấp xỉ bằng SigLIP2-base vision tower (~93 M) mà nó thay thế. Nghĩa là **đổi ngang về bộ nhớ**, được cái độc lập.

**Vướng:** cần augmentation (ràng buộc 2). BYOL/DINO không có augmentation sẽ collapse ngay — teacher và student thấy cùng input thì mục tiêu thành hằng số. Nên phương án này **phụ thuộc vào việc xây pipeline augmentation trước**, và nó cũng là phương án phức tạp nhất (cần EMA schedule, centering/sharpening để chống collapse).

**Nhận xét:** đây là phương án "đúng sách" nhất nhưng tốn công nhất. Không nên là bước đầu.

---

### Phương án D — Đổi teacher khác (DINOv3, v.v.)

Rẻ nhất về công (~vài dòng), và có lý do kỹ thuật thật: DINOv2/v3 nổi tiếng về **dense feature** chất lượng cao, phù hợp hơn SigLIP cho supervision mức patch — điều mà MAVT đang thiếu hoàn toàn (H17: chỉ có 1 vector toàn cục cho 10 240 số latent).

Nhưng **vẫn là phụ thuộc model ngoài**, nên không trả lời được câu hỏi của bạn nếu mục tiêu là độc lập. Chỉ nên coi là phương án tạm nếu mục tiêu thật là "semantic tốt hơn" chứ không phải "không phụ thuộc".

---

## 3. Đề nghị: kết hợp A + B, theo 3 bước

Semantic không cần đến từ một nguồn duy nhất. Cấu trúc content/detail của MAVT cho phép gắn hai objective ở hai chỗ khác nhau.

| Bước | Việc | Điều kiện tiên quyết |
|---|---|---|
| **0** | Sửa cache adjacency → key theo hash(positions); bỏ `no_grad` cho slot_diversity; dùng μ ở eval; lọc video hỏng | (đều là fix nhỏ đã nêu trong audit trước) |
| **1** | **B trên video + 3D**: cosine giữa content slot của 2 view + VICReg variance-covariance chống collapse | bước 0 |
| **2** | **A cho cả 3 modality**: mask 40–60% token, predictor đoán biểu diễn vùng che; áp lên content slot (ngữ nghĩa) và detail token (cục bộ) | bước 0 (cache!) |
| **3** | Đo, rồi mới quyết có cần C hay không | có baseline từ 1–2 |

**Vì sao thứ tự này:** bước 1 rẻ nhất và cho tín hiệu semantic *đúng bản chất* cho hai modality yếu nhất hiện nay. Bước 2 là phương án có tiềm năng cao nhất và là cái duy nhất phủ được ảnh. Bước 3 chỉ làm nếu 1+2 chưa đủ — tránh xây augmentation pipeline + EMA machinery trước khi biết có cần.

**Cách đo tiến bộ khi không còn teacher:** đây là điểm cần chuẩn bị trước. Bỏ SigLIP thì `cos_sim_teacher` không còn nghĩa, mà `metrics.py` hiện **không có** zero-shot classification hay retrieval (H18). Cần ít nhất một trong hai:
- **linear probe** trên một tập có nhãn (rẻ, đủ để so sánh giữa các phương án);
- **k-NN classification** trên content slot (còn rẻ hơn, không cần train).

Không có thước đo thì không phân biệt được "semantic đang hình thành" với "collapse êm đềm" — và với các objective không có negative thì collapse là chế độ lỗi mặc định.

---

## 4. Ba điều cần cảnh báo rõ

1. **Bỏ SigLIP distillation mà giữ SigLIP init thì vẫn là phụ thuộc SigLIP.** Cần quyết định mục tiêu thật là gì: "độc lập hoàn toàn" (khó, cần data lớn) hay "không cần teacher lúc train" (khả thi, các phương án trên đáp ứng).
2. **Số 0.815 hiện tại đang bị nhiễu sampling chi phối** (hai lần forward lệch cosine 0.705). Sửa việc này trước, rồi đo lại — có thể distillation vốn đã tốt hơn tưởng, và quyết định "bỏ SigLIP" đang dựa trên một con số sai.
3. **Mọi phương án tự-giám-sát ở đây đều không có negative** (trừ contrastive vốn đã bị chặn bởi batch), nên **collapse là chế độ lỗi chính**. Cơ chế chống collapse duy nhất trong repo (`slot_diversity_loss`) hiện là no-op. Đây là fix bắt buộc, không phải tuỳ chọn.

---

*Script tái tạo mọi con số: `audit4.py` → `audit4.json`.*
