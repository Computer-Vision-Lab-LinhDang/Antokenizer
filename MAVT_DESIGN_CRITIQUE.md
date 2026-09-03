# MAVT — Xét lại ở tầng thiết kế

*Không phải danh sách bug. Câu hỏi ở đây: các **tiền đề** của thiết kế có đứng được không. Đo trên `atoken/src/mavt/` HEAD `87dbdcf` bằng `audit5.py`.*

---

## 0. Kết luận trước, lập luận sau

Bạn nghi ngờ đúng chỗ. Nhưng vấn đề không phải "hướng này chưa đủ thông minh" — mà là **hai trong ba điểm đổi mới của MAVT, khi viết ra thành công thức, tự triệt tiêu**, và điểm thứ ba giải một bài toán mà cộng đồng đã có công cụ rẻ hơn 589 lần.

Cụ thể:

| Điểm đổi mới | Ý định | Điều thực sự xảy ra |
|---|---|---|
| Content-Detail Split | tách ngữ nghĩa / tần số cao | ở default xuất xưởng, **rút gọn thành `[slots(x) ; Affine(x)]`** — không có phép tách nào |
| RGAT4D | tiêm prior hình học có kiểu | tốn **589×** tham số so với relative-position-bias cho cùng tầm với; 3/4 loại cạnh rỗng ở ảnh |
| Understanding head sau z | buộc latent giữ cả hai | **đúng đắn** — nhưng số của chính AToken cho thấy đây là *thuế*, không phải cộng hưởng |

Đây là phát hiện ở tầng khác với audit trước. Audit trước nói "code có bug"; phần này nói "kể cả sửa hết bug, hai cơ chế này vẫn không làm điều bạn muốn".

![Bốn tiền đề thiết kế của MAVT khi đưa ra kiểm chứng bằng số]({{artifact:art_2d8f72f9-df9a-4cc7-9aa4-cd93056f4d64}})

---

## 1. Content-Detail Split tự triệt tiêu về mặt đại số

### Chứng minh

Ở `local_detail_window_size = 1` (default code, không config nào override):

```
grouped[:, i] = positions[:, i] // 1 = positions[:, i]
```

→ mỗi token là một group riêng → `mean-pool` trên **một** phần tử = **identity**.

Nhánh detail do đó rút gọn thành:

```
D = detail_proj(detail_norm(R))     # LayerNorm + Linear, không có pooling
```

Đo trực tiếp (`P1`):

| | giá trị |
|---|---|
| số token nguồn trên mỗi cửa sổ | **1.0** |
| `detail_branch == Affine(residual)`? | **đúng**, sai số tuyệt đối max = **0.0** |
| `residual_ratio` | 0.992 |
| cosine(detail_branch, `Affine(x_gốc)`) | **0.998** |

Cộng hai điều lại: vì `R ≈ x` (0.99, do bug chuẩn hoá trục — H2 audit trước), nhánh "chi tiết" thực chất là **một phép affine trên chính đầu vào**, cosine 0.998.

Nên "Content-Detail Split" hiện tại là:

```
compressed = [ slots(x) ; LayerNorm+Linear(x) ]
```

Không có phép tách. Không có phân rã tần số. Chỉ là **một bản sao affine của input nối vào sau các slot**.

### Vì sao đây là lỗi thiết kế, không phải lỗi code

Có thể lập luận rằng "sửa `dim=-1`→`dim=1` và đặt `window=2` là xong". Nhưng chú ý: hai lựa chọn đó **đã được bảo vệ có ý thức** trong `SLOT_ATTENTION_DETAIL_BRANCH_REPORT.md`, với lập luận rằng pooling residual lần hai sẽ "tái tạo lại chính mất mát mà residual sinh ra để tránh". Lập luận ấy đúng ở mức cục bộ và **sai ở mức hệ thống**: nếu nhánh residual không được nén thì không có tokenizer nào cả.

Đây là điểm cần đối diện ở tầng thiết kế: `window=1` không phải hằng số chọn nhầm, nó là **hệ quả tất yếu** của việc định nghĩa detail là "residual đầy đủ, có neo vị trí". Muốn có nén thì phải bỏ một trong hai tính chất đó.

---

## 2. Ngân sách token bị đảo ngược so với lượng thông tin

Đo trên **ảnh GT thật** (`results/infer_demo_stage1_3/image_input.png`), dùng SVD rank-N_c làm **chặn trên** cho những gì slot attention có thể đạt (`P3`):

| | nhánh content (64 token) | nhánh detail (256 token) |
|---|---:|---:|
| % ngân sách token được cấp | **20.0%** | **80.0%** |
| % năng lượng tín hiệu thực sự phải tải | **99.6%** | **0.4%** |

Thiết kế cấp 4× token cho nhánh tải 0.4% năng lượng. Đây là phân bổ nghịch. Kể cả sau khi sửa mọi bug, tỉ lệ 0.25/0.25 vẫn sai hướng — nó nên là ngược lại, hoặc detail nên rẻ hơn nhiều theo chiều kênh (latent nhỏ hơn) thay vì đông theo chiều token.

**Điểm rẽ nhánh thật:** phân bổ tối ưu phụ thuộc metric. Nếu mục tiêu là rFID/LPIPS thì 0.4% năng lượng đó lại chính là phần mắt người nhìn thấy — nên "năng lượng thấp" không có nghĩa "không quan trọng". Nhưng chính vì thế, quyết định này phải dựa trên một ablation đo được, không phải một tỉ lệ đặt trước.

---

## 3. Tiền đề "residual = tần số cao" chỉ đúng một phần

Đây là tiền đề nền của toàn bộ C-D split. Kiểm bằng phổ radial trên ảnh thật, với content branch **tối ưu** (SVD):

| N_c | năng lượng content giữ được | % residual ở **tần số cao** | % residual ở tần số thấp |
|---:|---:|---:|---:|
| 16 | 96.8% | 5.5% | 18.2% |
| 32 | 98.6% | 13.7% | 4.5% |
| **64 (cấu hình thật)** | **99.6%** | **31.4%** | 1.6% |
| 128 | 99.9% | 58.6% | 0.6% |

Ở cấu hình thật, **69% năng lượng residual không phải tần số cao**. Nói cách khác: dichotomy "content = ngữ nghĩa/tần số thấp, detail = texture/tần số cao" không phải thứ mà phép toán tạo ra — nó là thứ ta *mong* phép toán tạo ra. Residual của một xấp xỉ hạng thấp là **mọi thứ hạng thấp không mô tả được**, gồm cả cấu trúc trung tần và các thành phần hạng cao nhưng trơn.

Chỉ khi content branch rất mạnh (rank 128 = 50% số token) thì residual mới thực sự thiên về tần số cao — nhưng lúc đó ta đã tiêu hết ngân sách nén.

*(Cảnh báo: đo trên patch pixel thô vì không có checkpoint backbone đã train ở đây; feature đã train sẽ có phổ khác. Đây là chỉ dấu, không phải kết luận. Nhưng chiều của kết luận — dichotomy không sạch — khó đảo ngược.)*

---

## 4. RGAT giải một bài toán đã có công cụ rẻ hơn nhiều

RGAT4D tiêm quan hệ hình học có kiểu. Nhưng cách chuẩn trong lĩnh vực để tiêm cùng thông tin đó là **relative position bias** — một bảng tra cứu theo offset toạ độ.

| | tham số |
|---|---:|
| RGAT overhead so với block thường | **3 535 936** |
| Bảng relative position bias 4D cùng tầm với (r_s=2, r_t=1, 16 head) | **6 000** |
| **tỉ lệ** | **589×** |

Và phần lớn bộ máy đó không được dùng:

| modality | spatial | temporal | depth | cross-plane |
|---|---|---|---|---|
| image | dùng | rỗng | **rỗng** | rỗng |
| video | dùng | dùng | **rỗng** | rỗng |
| threed | dùng | dùng | **rỗng** | dùng |

`depth` (edge type 2) là **zeros tensor cho mọi modality** — chưa từng có một cạnh nào — nhưng vẫn cấp `k_proj` + `v_proj` riêng. Ở stage 1 (chỉ ảnh), **3.54 M tham số K/V chết**, không nhận gradient nào.

RGAT chiếm 21.2 M = **23% backbone**. Đây là chi phí lớn nhất trong kiến trúc cho thành phần có đóng góp chưa được chứng minh bằng bất kỳ ablation nào trong repo.

**Điểm công bằng cho RGAT:** khác với relative-position-bias, RGAT có K/V *riêng theo loại cạnh*, tức biểu đạt mạnh hơn về nguyên tắc — nó có thể học biến đổi giá trị khác nhau theo quan hệ, không chỉ lệch điểm attention. Câu hỏi thiết kế không phải "RGAT có mạnh hơn không" mà **"mạnh hơn có xứng 589× tham số và 23% backbone không"** — và câu đó chỉ trả lời được bằng ablation, thứ hiện chưa có.

---

## 5. "Unified" là thuế, không phải cộng hưởng — và MAVT đang đóng thêm thuế

Đây là điểm quan trọng nhất, và nó đến từ **chính số liệu AToken** mà `results.md` đã tổng hợp:

| Stage | rFID ảnh (thấp = tốt) | zero-shot ImageNet % (cao = tốt) |
|---|---:|---:|
| 1 | 0.258 | **82.7** |
| 2 | 0.246 | 82.3 |
| 3 | **0.209** | 82.2 |

Tái tạo tốt dần đều. Ngữ nghĩa **xấu dần đều**. Teacher SigLIP2 đạt 83.4 — và AToken không bao giờ vượt teacher, chỉ tiến gần rồi lùi ra.

Ở video còn rõ hơn: AToken 40.2 MSRVTT R@1 so với VideoPrism-g **52.7** — encoder không phải tái tạo gì cả. Khoảng cách **12.5 điểm** chính là giá của việc gánh thêm mục tiêu tái tạo.

`results.md` đọc bảng này theo hướng lạc quan ("degradation qua các stage rất nhỏ... chứng minh joint training không phá hủy semantic priors"). Cách đọc đó bỏ qua tính **đơn điệu**: ba điểm cùng chiều không phải nhiễu, đó là một trade-off curve.

**Hệ quả cho MAVT:** thiết kế của MAVT thêm một nhánh detail chuyên trách pixel — tức **đẩy mạnh hơn nữa** về phía đầu recon của trục đánh đổi này. Nếu số liệu AToken đúng, MAVT sẽ có recon tốt hơn và semantic **xấu hơn** AToken. Đó có thể là một lựa chọn hợp lý — nhưng phải là lựa chọn *có ý thức*, và hiện tại nó chưa được phát biểu ở đâu.

---

## 6. Dung lượng model so với dữ liệu: chế độ ghi nhớ

| | mẫu | tham số / mẫu |
|---|---:|---:|
| hiện tại (image10k) | 10 000 | **19 576** |
| nếu có ImageNet-1k | 1 281 167 | 153 |

195.8 M tham số trên 10k ảnh. Một tokenizer phải **generalize** — đây là chế độ ghi nhớ. Mọi kết luận về kiến trúc rút ra ở quy mô dữ liệu này đều không đáng tin: không phân biệt được "cơ chế này tốt" với "cơ chế này có nhiều tham số hơn".

Đây là lý do tôi nghĩ **câu hỏi thiết kế nên được trả lời sau, không phải trước**: hiện chưa có bàn thí nghiệm nào đủ tin để phân xử giữa các lựa chọn kiến trúc.

---

## 7. Ba hướng thiết kế lại, và cái giá của mỗi hướng

### Hướng A — Bỏ tham vọng unified, làm một tokenizer tái tạo tốt

**Luận đề:** cạnh tranh trên nén + rFID/rFVD. Bỏ understanding head, bỏ distillation. Semantic để encoder khác lo.

- Bỏ: understanding decoder (17.2 M), teacher, toàn bộ vướng mắc semantic
- Giữ: C-D split (sau khi sửa để thật sự nén), decoder
- Đo bằng: rFID, rFVD, PSNR/LPIPS ở tỉ lệ nén cố định
- **Được:** một mục tiêu duy nhất, đo được, không đánh đổi. Trục cạnh tranh rõ.
- **Mất:** không còn là "unified", tức mất điểm mới lạ so với các VAE tokenizer đã có (FLUX, Wan2.2). Phải thắng bằng số thuần.

### Hướng B — Giữ unified, nhưng chọn phía đánh đổi một cách công khai

**Luận đề:** chấp nhận trade-off curve ở mục 5, và tuyên bố MAVT nằm ở đâu trên đó — ví dụ "recon tốt hơn AToken 20% với cùng semantic", hoặc ngược lại.

- Giữ: understanding head sau z (điểm thiết kế đúng đắn nhất hiện có)
- Cần thêm: supervision mức patch (hiện chỉ có 1 vector toàn cục cho 10 240 số latent)
- Đo bằng: **cả hai trục cùng lúc**, vẽ ra curve chứ không báo cáo một điểm
- **Được:** giữ được tính mới lạ; trade-off curve tự nó là một đóng góp nếu vẽ được sạch
- **Mất:** cần cả hai bộ eval (rFID + zero-shot/retrieval), tức phải xây phần đo còn thiếu hoàn toàn

### Hướng C — Đổi trục cạnh tranh sang thứ chưa ai làm tốt: 3D

**Luận đề:** image/video tokenizer là chỗ đông đúc và MAVT đang kém 11 dB. Nhưng **3D tokenizer thì gần như trống**. `results.md` không có nổi một bảng baseline 3D nào để so.

- Bỏ triplane (audit trước: 64% cạnh hình học là giả) → chuyển sang voxel thưa hoặc point cloud, nơi toạ độ 4D của RGAT **thật sự có nghĩa**
- RGAT lúc này mới đúng chỗ: quan hệ hình học thật, `depth` edge type không còn rỗng
- **Được:** RGAT từ "chi phí chưa chứng minh" thành "lý do tồn tại"; cạnh tranh ở nơi thưa người
- **Mất:** phải đổi data pipeline; mất khả năng dùng SigLIP prior (không có teacher 2D cho voxel); rủi ro cao nhất

**Nhận xét của tôi:** hướng C là hướng duy nhất mà **RGAT — thành phần đắt nhất và đặc thù nhất của MAVT — trở nên có lý**. Ở image/video, prior hình học của nó gần như trùng với những gì relative-position-bias làm được với 1/589 tham số. Nếu bỏ 3D thì nên bỏ luôn RGAT và thu lại 23% backbone.

---

## 8. Điều tôi khuyên làm trước khi chọn hướng

Không nên chọn hướng bằng lập luận. Ba việc sau rẻ và sẽ phân xử hộ:

1. **Ablation nhỏ nhất có thể, trên cùng một tập:** `[không C-D split]` vs `[C-D split window=1]` vs `[C-D split window=2]` vs `[không RGAT]`. Bốn run ngắn (~10k step). Đây là bốn con số hiện đang thiếu, và không có chúng thì mọi tranh luận thiết kế đều là suy đoán.
2. **Xây phần đo trước khi xây model:** rFID + linear probe. Không có thước thì không phân biệt được tiến bộ với nhiễu — và ở quy mô 10k ảnh, nhiễu lớn.
3. **Quyết định quy mô dữ liệu trước khi quyết định kiến trúc.** 19 576 tham số/mẫu nghĩa là bàn thí nghiệm hiện tại không phân xử được gì. Đây là ràng buộc gốc; kiến trúc là thứ yếu so với nó.

---

## 9. Trả lời trực tiếp câu hỏi của bạn

"Hướng hiện tại có thông minh không?" — Câu trả lời tách làm ba:

- **Understanding head sau z: thông minh.** Giữ.
- **Content-Detail Split: ý tưởng đúng, hiện thực tự triệt tiêu.** Tiền đề "residual = tần số cao" chỉ đúng 31%. Cần định nghĩa lại, không phải sửa hằng số.
- **RGAT4D: đắt và chưa chứng minh ở image/video.** Chỉ có lý nếu chuyển sang 3D thật (voxel/point cloud).

Điểm yếu lớn nhất **không nằm ở kiến trúc**: nó là 10k ảnh, 52% video hỏng, và không có bộ đo nào cho các chỉ số quyết định. Một kiến trúc tốt hơn trên nền đó vẫn không cho kết quả đáng tin. Nếu chỉ sửa được một thứ trong tháng này, tôi sẽ chọn dữ liệu + thước đo, không phải model.

---

*Số liệu: `audit5.py` → `audit5.json`. Kết hợp với `MAVT_ARCH_HOLES.md` (tầng thực thi) và `MAVT_SEMANTIC_WITHOUT_SIGLIP.md` (đường semantic).*
