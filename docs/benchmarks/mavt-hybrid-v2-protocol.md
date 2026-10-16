# MAVT hybrid v2 — benchmark protocol

Cập nhật: **2026-10-17**. Trạng thái: **correctness reference đã kiểm; chưa train hybrid**.

[Thiết kế](../plans/mavt-hybrid-v2-design.md) · [Architecture](../architecture/mavt-hybrid-v2.html) · [API](../reference_hybrid_cd_model.md) · [Architecture preset](../../configs/mavt4d/hybrid_cd2_kv_v2.json)

## 1. Phạm vi và mức độ khóa

Protocol phân biệt ba lớp bằng chứng, không chuyển kết quả của lớp này thành
claim cho lớp khác:

| Lớp | Trạng thái | Chứng minh được gì |
|---|---|---|
| Hybrid technical gates | 83 hybrid tests qua; 576 CPU regression tests qua tại `20330de9afe278abee5694a2de65b1068ba95305` | Contract, causal execution, restart và parity trong phạm vi fixtures |
| Synthetic pilot v1 | Config/loss/evaluation đã khóa và có runner riêng | Một experiment C/D baseline có thể tái lập; không phải kết quả hybrid |
| Hybrid learning benchmark | Các câu hỏi, metrics và controls dưới đây đã xác định; còn gate freeze ở §10 | Chưa có learning/held-out/performance result |

[Preset hybrid JSON](../../configs/mavt4d/hybrid_cd2_kv_v2.json) chỉ khóa kiến trúc,
không phải training config. Không đưa preset này vào runner v1 rồi coi là hybrid
pilot đã được định nghĩa. Mọi thay đổi dataset, loss, metric, schedule hoặc
threshold sau kết quả phải có version/config hash mới, ghi lý do và báo cả số
cũ lẫn mới nếu đã đo. Không chọn checkpoint hoặc sửa metric theo held-out score.

## 2. Câu hỏi và dữ liệu kiểm soát

Ba câu hỏi chính:

1. C có giữ **thứ tự diễn biến đã quan sát** khi hai frame cuối giống hệt nhau?
2. D có giữ **configuration tương đối với event**, và decoder có dùng D trong
   khi C vẫn mang nội dung cần thiết?
3. C/D-only decoder đọc vùng thiếu tốt đến đâu khi evidence đã từng thấy, chỉ có
   ở view khác, hoặc chưa từng thấy? Mỗi điều kiện cần báo riêng.

Fixture hiện có là [event-cd-v1](../event_fixture_contract.md): mỗi source seed
được crossed với bốn core và hai configurations, tạo tám variants.

| Factor | Giá trị |
|---|---|
| Temporal core | `left_then_right`, `right_then_left`, `clockwise`, `counterclockwise` |
| Configuration | `compact`, `wide` |
| Source nuisance | Appearance/camera variation theo source, dùng chung cho tám variants |

Với cùng source và configuration, cả bốn core có **hai RGB frame cuối, positions,
timestamps và calibration giống hệt nhau**. Hai reversal pairs có cùng frame
multiset; lịch sử đi qua cùng poses theo thứ tự ngược lại. Configuration vẫn
quan sát được ở terminal state vì khác radius/depth. Audit phải kiểm exact
suffix equality và frame-multiset equality trước training.

Source group là đơn vị split và bootstrap; mọi core×configuration variant cùng
source ở cùng split. Fixture v1 dùng group hash `mavt-split-v1`, validation 0.2.
Không đưa source IDs, record paths, split, labels, GT boxes/positions/depth hoặc
future RGB vào model inputs. Inputs chỉ gồm RGB prefix, availability, timestamps
và calibration khả dụng. Targets/labels nằm ở kênh evaluation riêng.

Held-out ở đây là **sources mới**, gồm appearance/camera variation; không phải
held-out core×configuration combinations. Fixture v1 không có occluder thật,
multi-view recovery hoặc amodal targets. Thành công trên bốn motion labels
không tự chứng minh vai trò, tương tác, ý định hay bản chất event tự nhiên.

Hybrid long-lag suite phải có version/manifest mới; không sửa fixture v1 tại chỗ.
Renderer/loader phải audit input–target separation, paired masks, prefix storage
không giữ future pixels và source split trước khi model được chấm điểm.

## 3. Technical gates trước learning

Tham chiếu implementation: [`src/mavt4d/hybrid`](../../src/mavt4d/hybrid/).
Các tests không có optimizer steps; backward được dùng để kiểm gradient parity.

| Gate | Điều kiện kiểm | Nguồn test |
|---|---|---|
| Exact restart | Process mới, cùng weights/config, chỉ full C/D+KV+metadata; trước/sau ring wrap và missing | [test_hybrid_restart.py](../../tests/mavt4d/test_hybrid_restart.py) |
| Payload completeness | Strict fields/hash/dtype/shape/ring, compact snapshot, bytes, feedback identity | [test_hybrid_state.py](../../tests/mavt4d/test_hybrid_state.py) |
| Causality | Perturb future RGB, masks, timestamp, geometry; prefix states/readout không đổi | [restart tests](../../tests/mavt4d/test_hybrid_restart.py), [model tests](../../tests/mavt4d/test_hybrid_model.py) |
| Full/chunk/step/scan | Cùng tubelets, weights, initial state, accepted-step schedule; values và gradients | [model tests](../../tests/mavt4d/test_hybrid_model.py), [cell tests](../../tests/mavt4d/test_hybrid_attention.py) |
| KV/window | Current+15 previous steps; missing không kéo old keys vào bù; BF16 trước read | [attention tests](../../tests/mavt4d/test_hybrid_attention.py) |
| Missing/no-op | Missing Δt>0 age ring và zero-invalid KV; missing Δt=0 giữ toàn payload | [model tests](../../tests/mavt4d/test_hybrid_model.py) |
| Encoder isolation | Hidden NaNs/gradient bị chặn trước Patchify; padding, fixed camera identity, geometry riêng | [test_hybrid_io.py](../../tests/mavt4d/test_hybrid_io.py) |
| Decoder path | C=0→RGB=0, không KV input, finite C/D gradients, tile/pixel/query-chunk equivalence | [IO tests](../../tests/mavt4d/test_hybrid_io.py) |

Tolerances đã khóa trong các test tương ứng:

| So sánh | rtol | atol |
|---|---:|---:|
| Full/chunk/step/scan C/D và RGB trong raw pipeline | 3e-5 | 3e-6 |
| K/V giữa execution paths | 0.008 | 1e-4 |
| Metadata, no-op, các assertion isolation chính xác | 0 | 0 |
| Full temporal gradient parity | 0.03 | 2e-5 |
| Coupled cell scan values/gradients | 2e-5 | 2e-6 |
| Fresh-process payload/RGB continuation | 1e-5 | 1e-6 |

KV tolerance xử lý khác biệt rounding boundary giữa execution paths; cả hai
path vẫn bắt buộc dùng BF16 trước attention. Không dùng tolerance để hợp thức hóa
FP32 sequence attention so với BF16 streaming. Báo max absolute error, dtype,
initial state và lịch bước; không claim bitwise equality giữa mọi kernel.

Correctness hiện có chạy CPU. Default-width model được forward/decode trên grid
17×33; đó không phải throughput test 256×256. Observed Δt=0 được API chấp nhận
nhưng chưa có execution regression riêng; thêm gate này trước khi benchmark
workload dùng same-time assimilation. Mixed precision/module conversion và
optimized CUDA kernel cần correctness/parity riêng trước performance claims.

Regression 576 tests gồm 83 hybrid + 493 tests cũ trong phạm vi đã chọn. Một
optimizer smoke module và hai optimizer/checkpoint tests được bỏ để không chạy
train; server-data tests và phần tokenizer ngoài phạm vi này chưa được xác nhận.

Focused correctness command, chạy từ repo root trong môi trường đã có dependency:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest \
  tests/mavt4d/test_hybrid_state.py \
  tests/mavt4d/test_hybrid_io.py \
  tests/mavt4d/test_hybrid_attention.py \
  tests/mavt4d/test_hybrid_model.py \
  tests/mavt4d/test_hybrid_restart.py -q -p no:cacheprovider
```

## 4. Frozen runnable baseline: synthetic pilot v1

Nguồn chuẩn là [protocol v1](../synthetic_cd_pilot.md) và
[config `cd-synthetic-pilot-v1`](../../configs/mavt4d/synthetic_pilot_v1.json).
Các giá trị dưới đây chỉ áp dụng **v1**, không tự chuyển sang hybrid.

| Setting | V1 đã khóa |
|---|---|
| Dataset | 128 sources, seeds 0–127; 1024 clips, 12 frames, 32×48, dt=1/12 |
| Cells | Coupled affine, two-scan C→D, GRU |
| C/D budget | C8×16 + D8×16 FP32 + FP64 timestamp = 1032 byte/stream |
| Observer/decoder | Embed 32, patch 4, D modulation 8 |
| Batch / update / queries | Tám variants của một source; hai frames/update; 256 RGB queries/update |
| Tiny overfit | First training source, 300 optimizer steps, seed 17 cho mỗi cell |
| Main | 1200 optimizer steps/cell/seed; seeds 17,29,43 |
| AdamW | lr 0.001, betas (0.9,0.95), weight decay 0.0001, grad clip 1 |
| Execution | FP32, sequential recurrence, final checkpoint only |
| Dev | 10% training sources qua `mavt-pilot-dev-v1`; fixed run không dùng để chọn checkpoint |

V1 match state bytes, examples, updates và query counts. **Không match parameter
count hoặc compute**: GRU flatten state có nhiều parameters hơn. Copy shared
observer/decoder/auxiliary-head initialization giữa cells; cùng RNG seed đơn lẻ
không đủ. Source/objective RNG tách khỏi model RNG. Báo parameter count và elapsed
time thay vì gọi đây là một matched-compute hoặc fused-kernel comparison.

V1 không dùng core/configuration labels hoặc pretrained teacher trong representation
training. Objective đã khóa:

\[
L=L_{observed}+L_{missing}+0.5L_{future}+L_{history}
  +0.1L_{invariance}+0.05L_{noncollapse}.
\]

- Observed/missing L1 normalize riêng theo số valid RGB channels; empty subset
  góp zero. Cả hai dùng cùng C/D-only decoder.
- Future L1 từ masked prefix đến frame 5; rollout không observation tới frames
  7 và 9. Future RGB chỉ là loss target.
- Final-C history head dự báo pooled differences `x[2]-x[7]`, `x[3]-x[6]`, mỗi
  phần 3×8×12; RMS normalization floor 0.02. Reversal đổi dấu target; common
  final tubelet không xác định được target này.
- Invariance dùng normalized 16-D semantic projection của C, cùng temporal và
  availability support; photometric gain [0.9,1.1], bias [−0.05,0.05]. Không ép
  toàn raw C bỏ visual content cần cho reconstruction.
- Noncollapse dùng std floor 0.1 và off-diagonal sample covariance, cả hai views
  differentiable. Công thức đầy đủ giữ ở protocol v1; đây không phải class loss.

Tiny overfit dùng clean observations. Diagnostic pass khi final/initial clean
reconstruction L1≤0.75 và history MSE/zero-predictor MSE≤0.8. Báo absolute errors
cùng ratios. Finite failure vẫn chạy fixed main budget cho tất cả cells, không
tặng thêm steps cho một cell; nonfinite run fail rõ ràng. Các thresholds chỉ
chứng minh optimization progress, không có nghĩa thuộc tập nhỏ hoàn hảo.

V1 ridge probes: train-only mean/std (floor 1e-4), λ=1.0, Gram `XᵀX/n`, FP64
solve, unpenalized intercept. Main seeds báo riêng, mean và sample standard
deviation; bootstrap source groups 1000 resamples, seed 31415, paired giữa cells.
Source-bootstrap CI điều kiện theo fitted model, không thay thế uncertainty
giữa training seeds. Không dùng validation labels cho fit, normalization hay
checkpoint/loss selection.

V1 runner [`run_synthetic_pilot.py`](../../scripts/mavt4d/run_synthetic_pilot.py)
yêu cầu CUDA cho production learning; tài liệu này không khởi chạy runner.
Không có kết quả learning hybrid được suy từ status runner hoặc từ tests v1.

## 5. Hybrid primary C metric: event history trong C

Freeze representation trước khi fit probes. Feature primary là flattened
**C1+C2**, không có D, KV, raw frames hoặc decoder hidden features. Báo C1 và C2
riêng như diagnostics. Primary score là mean binary direction accuracy của hai
reversal pairs, chance 50%. Four-way core accuracy là secondary: order-blind
feature có thể phân biệt lateral/loop mà vẫn không hiểu reversal.

Fit scaler/probe trên train sources; khóa hyperparameters trước held-out. Chính
sách train-only probe của v1 được giữ, nhưng feature geometry và toàn hybrid
run phải có config/version mới. V1 ridge/bootstrap constants không tự là một
hybrid training lock; các giá trị hybrid được ghi rõ ở gate §10.

Required controls dùng cùng source groups và queries:

- **Last-tubelet/reset-state:** reset full S trước common final tubelet. C/D
  reset mà giữ KV không phải control này, vì KV còn lịch sử.
- **Order-invariant:** temporal mean của single-frame observer features, không
  time-position input. Sorting từng feature coordinate trước averaging loại
  rounding cues từ thứ tự cộng; báo feature dimension khác C/D nếu có.
- **Untrained matched initialization:** recurrent features của model chưa train,
  với cùng architecture/inputs. Above-chance probe không đủ chứng minh đã học
  nếu random recurrent features cũng mang tín hiệu đó.
- **D-only direction/core probe:** diagnostic xem event information tập trung ở
  đâu; không ép D hoàn toàn không được chứa temporal information.

Matched-suffix và orderless controls phải ở chance trên balanced reversal task.
Nếu chúng vượt chance, audit fixture/input/probe leakage trước khi diễn giải C.
Không chọn winner từ một seed hoặc chỉ báo một aggregate 4-way score.

### Long-lag bridge và can thiệp memory

Hybrid suite mới đặt distinguishing cue ở đầu, sau đó **64 logical steps có
observations giống nhau giữa paired histories**, rồi đọc/probe C. Hai windows
16 bước xếp tầng có attention-only receptive field khoảng 31 bước. Vì vậy
lag>16 không đủ làm long-memory diagnostic.

KV2 chứa lower-state readout: token cache mới có thể mang evidence rất cũ.
Lag64 một mình cũng không chứng minh thông tin chỉ tồn tại trong C/D. Khóa cut
ngay sau cue, final probe sau 64 bridge steps, với các interventions:

| Intervention tại cùng cut | Giữ / đổi | Câu hỏi |
|---|---|---|
| Intact | Giữ toàn S | Reference accuracy với cùng bridge |
| Flush cả KV | Giữ C/D, clock và logical schedule; zero K/V và token validity | C/D carry được gì sau khi cache content ban đầu bị loại? |
| Zero C/D, keep KV | Giữ cache và metadata | Cache có thể tái nạp event information vào C/D tới đâu? |
| Zero C/D và flush KV | Giữ cùng continuation schedule | Reset-memory control |
| Flush KV ngay trước decode | Giữ C/D/query | RGB hiện tại phải bất biến theo decoder contract |

Flush-after-cue chỉ là can thiệp **một lần**, không đồng nghĩa cache-disabled
throughout. Zero-C/D/keep-KV không được mặc định phải về chance: cache có thể
khôi phục state. Flush-before-decode là technical path check, không là evidence
semantic. Interventions phải giữ ring validation hợp lệ, cùng cut/times/inputs;
không tùy chọn cut sau kết quả. Lag1/8/16/32/64 curves là secondary diagnostics.

## 6. D giữ configuration và decoder dùng cả C/D

Primary D score là configuration balanced accuracy từ frozen **D1+D2**, tổng
và theo từng core. Per-core scores stratify cùng globally fitted probe, không
fit một probe mới cho mỗi core. Báo D1/D2 riêng; compact/wide chỉ kiểm hai
configurations này, không chứng minh phase, velocity hoặc disentanglement.

Cross-probes C→configuration và D→core/direction kiểm information overlap.
Không yêu cầu các score chéo bằng zero; điều cần phát hiện là C bỏ diễn biến
trong khi D mang gần hết task. Báo cả primary C/D scores và cross-probes.

Decoder interventions gồm D zero và D swap ở paired counterfactuals có geometry,
query và timestamp tương thích; giữ C cố định, đo reconstruction/configuration
readout. Chỉ diễn giải swaps có supported target trong fixture, không dùng
off-distribution latent swaps tùy ý như bằng chứng disentanglement. C zero là
assertion content path: RGB phải zero chính xác; nó không tự chứng minh C đã học
bản chất event. Cũng không suy cấu trúc D chỉ từ việc probe giải mã được label.

## 7. Anisotropic masking, missing readout và inference

Mask được lấy độc lập RGB, core/configuration labels và target geometry; tất cả
variants của một source dùng cùng sampled mask. Áp mask trước Patchify. Clean
target/teacher chỉ ở loss, không vào student state, K/V hoặc query selection.
Photometric consistency phải dùng cùng temporal/availability support để tránh
ép C bỏ evidence quan trọng chỉ vì một augmentation xóa mất event cue.

Frozen v1 mask families có probability 0.2 mỗi loại:

| Family | V1 train geometry/time |
|---|---|
| clean | Toàn observation khả dụng |
| wide | Rectangle H/2×W/2 thiếu trong một hoặc hai tubelets |
| tube | Rectangle H/2×W/4 thiếu từ frame 4 trở đi |
| snapshot | Toàn frames 4–7 thiếu |
| never_seen | Rectangle H/2×W/2 thiếu suốt clip |

Đây là các extent khác nhau theo không gian/thời gian, không chỉ random pixel
dropout. Hybrid có tubelets/long-lag suite mới: giữ các family và nguyên tắc
independence, nhưng spatial/time extents, probabilities và curriculum phải được
freeze trong config hybrid mới trước run. Chưa có trainer anisotropic hybrid
được tích hợp. Không coi các con số v1 phía trên là ngầm khóa cho video tự nhiên.

Report L1 và PSNR riêng trên observed/missing RGB channels, theo family và lag.
RGB target range [−1,1], PSNR data range=2, aggregate squared error, không clipping.
Báo channel counts, không trung bình thiếu kiểm soát các per-image PSNR. Empty
region có null metric; zero error có null PSNR và `perfect=true`, không numerical
cap tùy ý. LPIPS không có trong frozen v1 pilot.

V1 decoder evaluation dùng bốn sorted held-out sources đầu, mask seed 2718.
Snapshot đọc tại frame7 khi vẫn missing; các family còn lại tại frame11, wide
evaluation mask phải phủ endpoint đó. Hybrid evaluation sources/query counts
và endpoints phải khóa riêng theo suite mới; không lựa vùng dễ sau kết quả.

Điều kiện inference cần tách rõ:

- **Từng thấy rồi thiếu:** history có evidence; đo reconstruction theo lag và
  exposure, so với copy-last cùng vùng/query/lịch bước.
- **Chỉ thấy ở view khác:** cần multi-view fixture và calibration thật; fixture
  single-view v1 chưa đo trường hợp này.
- **Chưa từng thấy:** báo riêng; output là inference từ prior, không gọi là
  memory recovery. Fixed image rectangle `never_seen` chưa tự chứng minh cùng
  một 3D surface chưa từng quan sát do camera/object motion.
- **Toàn bước thiếu:** cùng cell rollout, không synthetic observation write;
  age KV theo accepted steps, decode sau rollout. Các baseline dùng cùng lịch.
- **Occluder thật:** RGB của occluder vẫn là valid observation. Availability
  không phải oracle object-visibility mask; RGB pixel query tái tạo visible
  surface. Amodal/object/depth-layer readout cần query và target mới.

## 8. Baselines và công bằng ngân sách

Mỗi run báo readout-code bytes, full-restart bytes, parameter count, peak
inference memory, FLOPs hoặc latency, throughput và task quality. Hybrid default
có 16 KiB C/D nhưng full stream tensors **148256 byte**, thêm 68 byte header/payload.
Không gọi nó memory-matched với baseline 16 KiB chỉ vì C/D bằng nhau.

| So sánh | Cần giữ | Cần công khai |
|---|---|---|
| Same-CD ablation | C/D 4096 FP32 values | Full memory khác nhau khi bật/tắt cache |
| Same-full-memory | Cùng trần causal bytes, gồm cache/metadata | Width/state layout có thể khác |
| Same-training-compute | Samples/updates/tokens và compute policy đã khóa | Parameters/state bytes, actual elapsed/FLOPs |
| Cell substitution | Cùng observer, decoder, GQA skeleton; thay coupled/two-scan/GRU modules | Không tự là attention ablation hoặc matched parameters |

Hybrid ablation plan: cache disabled; một stage so với hai; direct intra-cell
`a_CD` off với common normalizer; matched untrained model; C/D-only decoder trong
mọi nhánh so sánh. Config hiện tại cố định hai stages; single-stage và alternative
cell variants cần implementation/config mới trước run.

`feedback=False` chỉ cắt D trước→C mới bên trong từng cell. Đường
**D1→mixer→GQA2→C2 vẫn tồn tại**. Không gọi đây là loại mọi D→C feedback.
Full-feedback-cut ablation cần spec/interface/normalization riêng trước đo.

Đo latency toàn pipeline: spatial encoder, GQA/window/cache update, scans,
decoder, và serialization khi workload cần. Báo batch/sequence length, physical
step schedule, query count, dtype, warmup và hardware class. Attention workspace,
weights và training activations nằm ngoài persistent state nhưng vẫn thuộc peak
memory. Reference Python gathers và generic scan chưa có claim nhanh hơn GRU.

## 9. Quy tắc báo cáo

Mỗi learning result cần dataset/config/weights identity và source revision;
source-group splits, seeds, cell/ablation, final checkpoint policy; primary C/D
scores, per-core/direction/cross-probes và controls; observed/missing metrics với
counts; budgets và compute; failures/nonfinite runs. Report từng seed trước
mean/std, và paired source uncertainty theo policy đã freeze.

Không dùng validation cho normalization/probe fit hoặc chọn checkpoint. Không
chọn winner bằng một seed, throughput đơn lẻ hoặc score trên chance khi untrained
control đã đạt tương đương. Run chưa hoàn tất không có held-out conclusion.
Technical tests qua không phải learning result. Khi chuyển sang natural video,
source-disjoint splits và target leakage audits vẫn bắt buộc; không tạo C/D class
loss rồi gọi representation training là không nhãn.

## 10. Gate còn phải khóa cho hybrid learning benchmark đầy đủ

**Bổ sung trước train, 2026-10-17:** [hybrid synthetic pilot v2](../hybrid_synthetic_pilot_v2.md)
khóa một pilot ngắn riêng: fixture 12 frames render mới 128×192, một coupled
hybrid/seed, training objectives, DDP và recovery. Pilot đó kiểm optimization
và held-out short-history C/D; không thay thế long-lag suite, memory
interventions hay baseline matrix bên dưới. Không sửa hoặc tái diễn giải v1.

Các mục sau **chưa hoàn thành**; không có hybrid loss/threshold được ngầm chọn
trong tài liệu này:

| Gate | Nội dung cần có trước learning run |
|---|---|
| Dataset version | Long-lag renderer/manifest/checksums; exact suffix/multiset/source audits; cue cut và 64-step bridge |
| Training config | Source counts, seeds, train/dev/val assignments, resolution/frame schedule, batch/queries, steps, optimizer, precision, checkpoint policy |
| Objectives | Loss definitions/weights, C-history target cho suite mới, reconstruction/future terms, consistency/noncollapse choices và valid-mask normalization |
| Mask schedule | Anisotropic extents/probabilities/time support, held-out masks/endpoints; không dùng target-derived masking |
| Evaluation lock | Probe/scaler hyperparameters, source/seed bootstrap policy, primary/secondary metrics, practical success criterion nếu cần quyết định go/no-go |
| Tiny-overfit gate | Dataset size, fixed budget, diagnostic thresholds; không tự tái dùng v1 thresholds |
| Baseline matrix | Same-CD/full-memory/compute grouping, valid one-stage/cache-off/cell variants, matching initialization và RNG |
| Runner/evaluator | Hybrid full-state checkpoint/restart, declared state interventions, masked losses, artifact recording; không đổi v1 runner nghĩa ngầm |
| Runtime correctness | Same-time assimilation test nếu workload dùng; CUDA/kernel parity nếu chạy GPU implementation khác |

Chỉ sau khi commit đầy đủ protocol/config/fixtures version mới mới chạy tiny
overfit rồi held-out hybrid benchmark. Kết quả lúc đó trả lời những câu hỏi ở
§2; publication này chỉ công bố thiết kế và protocol, không báo một learning
experiment chưa được thực hiện.
