# State hybrid: ba cách tổ chức pipeline (A, B, C)

Trạng thái: bản nháp để chốt kiến trúc, 16/10/2026. Kiến trúc đề xuất ở mục 8, quyết định cần chốt ở mục 9. Code tham chiếu: `src/mavt4d/model/state.py` trên nhánh
`work/state-synth` (commit `3956a05`).

## 1. Bối cảnh và những gì đã chốt

Buổi brainstorm ngày 15–16/10 đã chốt các điểm sau trước khi tới câu hỏi tổ chức module state:

| Câu hỏi | Đã chọn |
|---|---|
| Phạm vi | Cả pipeline đầu–cuối |
| State phục vụ ai trước | **Tokenizer tái tạo** (predictive model và 4D world model đến sau) |
| Dữ liệu bản đầu | Cả ảnh, video một camera và nhiều camera 4D, chung một mô hình |
| Mã mỗi bước | zₜ = VAE(Sₜ), state đầy đủ, mỗi bước tự giải mã được |
| Hình state | K token tự do (không neo lưới hay voxel) |
| Trunk | DINOv3 ViT-B/16 (768 chiều), **đóng băng**, đặc trưng cache được; module state nhỏ train riêng |
| Ngân sách | K = 128 × số view, tức 4.096 float (K × 32) cho mỗi view 256², bằng SD-VAE và perceiver v3 |

Câu cuối, "chọn A, B hay C", bạn trả lời "cách khác" và chưa kịp mô tả. Tài liệu này dựng lại ba phương án,
tách chúng thành ba trục quyết định để bạn chỉ ra được phương án của mình nằm ở đâu, rồi so sánh.

Code hiện tại (đọc từ `state.py`):

```
S⁻ₜ = Sₜ₋₁ + predict(Sₜ₋₁, Δt)                         predict khởi tạo bằng 0, tức identity
Sₜ  = LN(blocks([patch bước t + hình học tia ; S⁻ₜ])[-K:])   nếu có view nhìn thấy; không thì Sₜ = S⁻ₜ
zₜ  = VAE(Sₜ)                                          KL tới N(0, I)
rgbₜ, log_depthₜ = readout(zₜ, hình học camera bước t)   query chỉ mang hình học, mọi nội dung đi qua zₜ
```

Stem hiện là `CameraStem` (conv patch, train từ đầu), K = 16 cố định, d = 256, 4 khối self-attention chung
cho patch và state. Đây chính là **phương án C** với stem train từ đầu.

## 2. Ba trục quyết định

Mỗi bước t, module state nằm giữa đặc trưng DINOv3 của bước t và zₜ. Ba câu hỏi phân biệt các cách tổ chức:

1. **Thứ tự bằng chứng vào state.** Prior S⁻ₜ được dựng trước rồi mới đọc bằng chứng (lọc kiểu Bayes:
   predict rồi update), hay bằng chứng và state trộn đồng thời trong cùng một khối.
2. **Chỗ hợp nhất nhiều view.** Trong không gian patch (trước khi nén), trong không gian state (sau khi nén),
   hay cả hai.
3. **Chỗ đặt cell.** Cell là phần quyết định giữ bao nhiêu của S⁻ₜ và lấy bao nhiêu bằng chứng mới (none, GRU,
   SSM, delta rule). Cell có thể nằm tách riêng sau association, hoặc không có cell tường minh mà residual
   của các khối attention đóng vai trò đó.

| | Thứ tự bằng chứng | Hợp nhất view | Cell |
|---|---|---|---|
| **A. Association rồi cell** | predict → cross-attn đọc bằng chứng → cell → self-attn giữa token state | trong state | tường minh, tách riêng |
| **B. Hợp nhất snapshot trước** | patch mọi view trao đổi trước → rồi như A | trong patch, rồi trong state | tường minh (như A) |
| **C. Perceiver lặp** (code hiện tại) | predict → [patch ; S⁻ₜ] qua chung các khối self-attn | patch và state cùng lúc | ngầm (residual) |

## 3. Ba phương án

### A. Association rồi cell

```
F_t  = DINOv3(view nhìn thấy của bước t) + hình học tia (nếu có pose)      đóng băng, cache
S⁻ₜ  = Sₜ₋₁ + predict(Sₜ₋₁, Δt)
Uₜ   = CrossAttn^L(query = S⁻ₜ, mem = F_t)                                  association, mask view ẩn
Sₜ   = Cell(S⁻ₜ, Uₜ)                                                         none | GRU | SSM | delta rule
Sₜ   = SelfAttn^M(Sₜ)                                                        hợp nhất quan hệ giữa token
```

- Đúng "hybrid theo vai trò" trong bản phân tích 15/10: attention lo liên kết bằng chứng, cell lo cập nhật theo
  thời gian.
- Mỗi biến thể cần so (GRU, Mamba/SSM, delta rule, association có hoặc không có hình học) chỉ là thay một khối.
- Patch không bị cập nhật, nên đặc trưng DINOv3 đóng băng dùng thẳng làm key/value, không cần lan truyền ngược
  qua phía patch.

### B. Hợp nhất snapshot trước, rồi association

```
F'_t = SelfAttn^P(F_t)                       patch mọi view của bước t trao đổi với nhau (có thể dùng hình học)
rồi như A với mem = F'_t
```

- Nhất quán giữa view có thể tốt hơn: vật thể nằm vắt qua hai camera được ghép trước khi nén.
- Với ảnh hoặc video một camera, khối này chỉ là tinh chỉnh không gian trên đặc trưng DINOv3, gần như thừa.
- Tốn nhất ở nhiều camera: nuScenes có 2.688 patch mỗi bước, attention đầy đủ là 7,2 triệu điểm mỗi lớp.

### C. Perceiver lặp (code hiện tại)

```
x  = [F_t ; S⁻ₜ]
x  = SelfAttn^L(x)            patch và state cập nhật lẫn nhau
Sₜ = LN(x[-K:])
```

- Gần code nhất: chỉ cần đổi stem sang DINOv3 và đổi K.
- Không có bước association hay cell riêng để thay hoặc bỏ; muốn thử GRU hay delta rule phải chen thêm khối.
- Patch cũng đi qua MLP và attention mỗi lớp, nên tốn hơn A, và lợi thế "đặc trưng đóng băng" bị mất một nửa:
  các khối trainable vẫn phải giữ activation cho toàn bộ N + K token.

## 4. Mô hình hoá bất định: 4D world model và 4D predictive model

Mục tiêu bản đầu là tokenizer tái tạo. Nhưng state về sau phải phục vụ hai mô hình có nhu cầu bất định khác
nhau, và phương án tổ chức quyết định chỗ nào cắm được nhu cầu đó.

**4D predictive model** (dự báo hình học và ngoại quan của các bước tới, không có hành động). Cần:
- một **prior học được** p(zₜ | S⁻ₜ) để dự báo nhiều bước khi không có quan sát;
- phân biệt được vùng "chưa từng thấy" và vùng "đã thấy, đang chắc", để sai số theo horizon tăng đúng chỗ.

**4D world model có bất định** (state + hành động → state kế tiếp). Cần thêm:
- hàm chuyển trạng thái **ngẫu nhiên** có điều kiện theo hành động, kiểu RSSM: phần tất định hₜ cộng phần
  ngẫu nhiên zₜ, train bằng KL(posterior ‖ prior);
- posterior và prior tách rời rõ ràng để KL đó có nghĩa.

Code hiện tại có VAE với KL tới N(0, I), tức bất định chỉ ở mức "nén", không có prior theo thời gian. Điều này
đúng như nhau cho cả A, B và C: **phần bất định chủ yếu nằm ở predict và ở prior head, không phải ở cách tổ
chức association**. Khác biệt giữa ba phương án là chỗ cắm có sạch hay không:

| | A | B | C |
|---|---|---|---|
| Prior S⁻ₜ và posterior Sₜ tách rời | Rõ: S⁻ₜ là input của cell, Sₜ là output | Rõ (như A) | Có S⁻ₜ, nhưng posterior trộn lẫn với cập nhật patch |
| Thêm prior head p(zₜ \| S⁻ₜ) và KL(posterior ‖ prior) | Một đầu tuyến tính trên S⁻ₜ | Như A | Như A, nhưng prior và posterior đi qua cùng khối nên dễ "rò" bằng chứng |
| Gain kiểu Kalman (tin bằng chứng bao nhiêu) | Cell có cổng tường minh; khối lượng attention mỗi token nhận được là một tín hiệu "đã thấy bao nhiêu" | Như A, cộng bằng chứng đã hợp nhất nên ít mơ hồ giữa view hơn | Ngầm trong residual, không đọc ra được |
| Vùng chưa thấy giữ nguyên bất định | Token không nhận bằng chứng thì cell giữ S⁻ₜ | Như A | Mọi token đều bị trộn với patch mỗi lớp |
| Điều kiện theo hành động (world model) | Đưa aₜ vào predict, không đụng association | Như A | Như A |
| Bất định giữa view (che khuất, vật thể vắt qua camera) | Giải quyết trong state | **Tốt nhất**: giải quyết trước khi nén | Giải quyết cùng lúc với nén |

Tóm lại: với predictive model, A và B cho chỗ cắm prior và gain rõ nhất. Với world model, cả ba đều đưa hành
động vào predict, nhưng chỉ A và B có cặp prior/posterior đủ sạch để KL kiểu RSSM có nghĩa. B chỉ hơn A ở bất
định giữa view trong cùng một bước.

## 5. Chi phí tính toán và bộ nhớ

Ước lượng thô cho phần module state mỗi bước (DINOv3 đóng băng, cache nên không tính), d = 768, A và C có 4 lớp
association/khối, A có thêm 2 lớp self-attn giữa token state, B thêm 2 lớp hợp nhất snapshot đầy đủ. K = 128 × V.
FLOPs đếm attention, chiếu q/k/v/o và MLP (tỉ lệ 4).

| Dữ liệu | N patch | K | A | B | C | B/A | C/A |
|---|---|---|---|---|---|---|---|
| Ảnh hoặc video 256², 1 view | 256 | 128 | 12,6 GF | 20,2 GF | 23,6 GF | 1,6× | 1,9× |
| nuScenes 6 view 256×448 | 2.688 | 768 | 112 GF | 233 GF | 342 GF | 2,1× | 3,0× |

Điểm attention mỗi lớp ở nuScenes: A 2,1 triệu (K × N), B riêng phần hợp nhất 7,2 triệu (N²), C 11,9 triệu
((N + K)²).

Bộ nhớ activation khi lan truyền ngược qua T bước tăng theo T × số token đi qua khối trainable:
- **A**: K token qua mọi lớp, cộng phần chiếu key/value của N patch. Khoảng 26 nghìn vector d mỗi bước ở
  nuScenes.
- **B**: A cộng N patch qua các lớp hợp nhất. Khoảng 42 nghìn.
- **C**: N + K token qua mọi lớp. Khoảng 41 nghìn, và tăng theo số lớp nhanh hơn A.

Với T = 8 bước nhiều camera trên một A100, A còn chỗ để không cần gradient checkpointing ở cỡ batch nhỏ; B và
C nhiều khả năng cần. Các con số trên là ước lượng giấy, chưa đo.

## 6. Thay đổi code cần làm

Chung cho cả ba (nên làm trước, độc lập với lựa chọn):
- Thay `CameraStem` bằng bộ trích đặc trưng DINOv3 ViT-B/16 đóng băng, có cache đặc trưng trên đĩa, cộng một lớp
  chiếu 768 → d.
- `StateConfig.n_state` đổi thành `state_per_view = 128`, K = 128 × V tính lúc chạy.
- `Attention` nhận thêm attention mask, để view ẩn bị loại khỏi key thay vì thay bằng `mask_token` (cần cho A
  và B, nơi state chỉ đọc patch qua cross-attention).
- Giữ nguyên các tính chất đã có test trong `tests/mavt4d/test_state_model.py`: nhân quả, bước bị bỏ và view
  ẩn không ảnh hưởng, readout chỉ qua z, predict là identity lúc khởi tạo.

Riêng từng phương án:

| | Module mới | Sửa `StateModel.forward` | Test thêm |
|---|---|---|---|
| A | `Associate` (L khối `Block(cross=True)`, đã có sẵn), giao diện `Cell` với 4 bản (none, GRU, SSM, delta), `StateMix` (M khối self-attn) | Thay vòng `for blk in self.blocks` trên chuỗi nối bằng ba bước | Mỗi cell giữ đúng S⁻ₜ khi không có bằng chứng; patch không nhận gradient |
| B | A cộng `SnapshotFusion` (P khối self-attn trên patch của bước t, tuỳ chọn có hình học) | Như A, thêm một bước trước association | Hợp nhất không trộn patch giữa các bước |
| C | Không | Chỉ đổi stem và K | Không |

Thêm cho mục tiêu dự báo về sau (không thuộc bản tokenizer đầu tiên): một prior head p(zₜ | S⁻ₜ) và KL(q ‖ p)
thay cho KL tới N(0, I); một đầu vào hành động cho `predict` khi có dữ liệu có hành động.

## 7. Đề xuất

**Làm A làm khung chính, giữ C làm nhánh đối chứng, để B làm nâng cấp có điều kiện.**

- A tách vai trò rõ nhất, nên mọi so sánh đã lên kế hoạch (cell GRU, SSM, delta rule; association có hoặc không
  có hình học) chỉ là thay một khối. Nó cũng là phương án rẻ nhất ở nhiều camera và cho chỗ cắm prior/posterior
  sạch nhất cho predictive và world model về sau.
- C đã có sẵn và đã chạy. Đổi stem sang DINOv3 rồi chạy C cạnh A trên cùng cảnh tổng hợp cho một phép so sạch
  "association + cell" với "perceiver lặp", trước khi bỏ C.
- B chỉ đáng thêm nếu A cho thấy nhất quán giữa view yếu, đo bằng điều kiện `hide_view` và `hide_edge` sẵn có
  trong `scripts/state/train_synth.py`. Khi đó nên thử hợp nhất có hình học (attention theo epipolar hoặc theo
  cửa sổ) thay vì attention đầy đủ N², để giữ chi phí gần A.
- Kết quả job DGX `14575` (có/không tín hiệu trí nhớ) không đổi đề xuất này, nhưng nếu nhánh "có" vẫn không nhớ
  được trên cảnh mới thì đó là lý do làm A sớm: cell tường minh là chỗ ép giữ trí nhớ dễ nhất.

## 8. Kiến trúc đề xuất để chốt

Bản này gộp đề xuất ở mục 7 với ba phát hiện của bản review plan MAVT (16/10):
- correspondence graph (hub, cạnh PROJ/TRACK) chưa được model nào dùng, trong khi cổng M1 đang trượt trên
  nuScenes-mini (15,83% false-visible so với ngưỡng < 5%);
- chữ "uncertain" chưa có cơ chế: KL tới N(0, I) trọng số 1e-4, `predict` tất định, loss L1, eval lấy `mu`;
- quan hệ giữa state và tokenizer stage 1 (`src/mavt`) chưa được định nghĩa.

### 8.1 Một bước t

```
Đầu vào   F_t   = DINOv3-B/16(view nhìn thấy)              đóng băng, cache; 768 → d
                  + emb(tia camera) nếu có pose, emb(không pose) nếu là ảnh/video thường
Prior     h⁻ₜ   = hₜ₋₁ + predict(hₜ₋₁, zₜ₋₁, Δt, aₜ)        aₜ = pose ego của bước t (world model)
          p(zₜ | h⁻ₜ) = N(μᵖ, σᵖ)                           prior head, mỗi token một Gauss 32 chiều
Associate Uₜ    = CrossAttn⁴(query = h⁻ₜ, key/value = F_t)   view ẩn bị loại bằng attention mask
Cell      hₜ    = GRU(h⁻ₜ, Uₜ)                                cổng = "tin bằng chứng bao nhiêu"
Mix       hₜ    = SelfAttn²(hₜ)                              hợp nhất giữa K token
Posterior q(zₜ | hₜ) = N(μ^q, σ^q)                          bước không có view nào: q := p
Mã        zₜ ~ q                                              K = 128 × V token × 32 chiều; đây là token
                                                              cho mô hình sinh, predictive và world model
Readout   rgbₜ, log_depthₜ = readout(zₜ, camera bước t)      giữ nguyên: query chỉ mang hình học
```

Trạng thái truyền qua bước là cặp (hₜ, zₜ) như RSSM: phần tất định h mang trí nhớ dài, phần ngẫu nhiên z mang
bất định. Vì `predict` nhận zₜ₋₁ đã sample, nhiều mẫu z cho ra nhiều tương lai khác nhau.

### 8.2 Loss

```
L = L1(rgb) + 0,5 · L1(log_depth)                  trên các ô được giám sát (memory signal "on")
  + β · max(free_bits, KL(q(zₜ | hₜ) ‖ p(zₜ | h⁻ₜ)))   thay cho KL tới N(0, I)
  + β₀ · KL(p(z₁) ‖ N(0, I))                        chỉ bước đầu, giữ thang của z
```

Bước bị ẩn không có posterior nên không có KL; readout đọc z sample từ prior và vẫn bị giám sát, đó chính là
tín hiệu trí nhớ đang có trong `train_synth.py`.

### 8.3 Đo bất định

Thêm vào eval của `train_synth.py`, trên các ô bị ẩn (`drop_last`, `drop_last2`, `hide_view`):
- best-of-8 so với trung bình 8 mẫu (PSNR trên pixel hộp): mẫu tốt nhất phải hơn trung bình rõ rệt thì mới có
  đa giả thuyết;
- độ trải giữa các mẫu tăng theo số bước bị ẩn liên tiếp, và gần 0 trên ô đã thấy;
- trên E1.3: clip `occlusion_reappearance` và `same_final_position_different_direction`.

### 8.4 Graph và tokenizer stage 1

- **Graph** chưa nối vào bản đầu. Chỗ cắm đã có sẵn trong A: cạnh PROJ thành attention bias trong Associate, hub
  thành token key/value thêm. Chạy thí nghiệm có/không graph trên cảnh tổng hợp (có GT visibility) trước khi
  tiếp tục kéo M1 trên nuScenes. Nếu graph không giúp ở `hide_view`/`drop_last2`, hạ ưu tiên M1.
- **Stage 1:** state thay tokenizer stage 1 cho dữ liệu 4D. DINOv3 đóng băng đã thay vai trò encoder RGAT; zₜ là
  token. Ảnh là chuỗi một bước, video là V = 1.

### 8.5 Thứ tự làm

1. Đổi stem sang DINOv3 đóng băng có cache, K = 128 × V, attention mask cho view ẩn (chung cho mọi phương án).
2. Chạy C (code hiện tại với stem mới) làm đối chứng.
3. Dựng A với cell GRU, prior/posterior và eval bất định ở 8.3; so với C cùng compute trên `hide_view`,
   `drop_last2`.
4. Thí nghiệm graph-vào-state trên A.
5. Cổng state trên nuScenes-mini: thắng `copy_prev` ở `drop_last` trên pixel hộp.

B chỉ thêm nếu bước 3 cho thấy `hide_view` yếu.

## 9. Quyết định cần bạn chốt

Mỗi câu trả lời một từ; khuyến nghị in đậm.

| # | Câu hỏi | Lựa chọn |
|---|---|---|
| 1 | Khung module state | **A** / B / C / khác |
| 2 | Cell đầu tiên trong A | **GRU** / none / delta |
| 3 | Prior/posterior kiểu RSSM ngay từ bản đầu | **Có** / không |
| 4 | Graph vào state | **Sau** (thí nghiệm ở bước 4) / ngay |
| 5 | State thay tokenizer stage 1 hay đọc token của stage 1 | **Thay** / đọc |
| 6 | Hành động aₜ cho world model | **Pose** ego / điều khiển |

Nếu câu 1 là "khác", phương án của bạn khác A ở trục nào trong mục 2 là đủ để mình viết lại mục 8.

Các câu nhỏ hơn đã lấy mặc định và có thể đổi sau: hình học tia dùng khi có pose, emb "không pose" khi không có;
L = 4 lớp Associate, M = 2 lớp Mix; DGX không có gói Mamba nên delta rule nếu chọn sẽ viết bằng PyTorch thuần.
