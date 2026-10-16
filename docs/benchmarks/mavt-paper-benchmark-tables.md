# MAVT — proposed paper benchmark tables

Last updated: **2026-10-17**. Status: **proposed, not frozen; no benchmark results**.

Tài liệu lưu cấu trúc **năm bảng chính và ba nhóm ablation** cho paper. Mọi ô
kết quả chưa đo đều là **—**. Các hàng là kế hoạch so sánh, không có nghĩa
baseline, adapter, dataset extension hoặc evaluator tương ứng đã được triển khai.

[Hybrid benchmark protocol](mavt-hybrid-v2-protocol.md) · [Design plan](../plans/mavt-hybrid-v2-design.md) · [Reference API](../reference_hybrid_cd_model.md) · [Documentation index](../README.md)

## Scope and reporting conventions

`Shared encoder` trong các controlled comparisons là cùng **Patchify + spatial
backbone**, giữ thiết kế và quy tắc initialization chung. Khác biệt temporal
backbone/state được khai báo riêng. Frozen-encoder evaluation dùng cùng task-head
family và budget; một baseline không bắt buộc có split C/D để tham gia so sánh.

Các nhóm controlled training, external pretrained checkpoints và kết quả trích
từ paper phải tách rõ. Không lấy published score từ training data/protocol khác
rồi trình bày như reproduced result cùng setting. Tất cả rows dưới đây chưa đo;
không có imported paper scores.

Khóa dataset version, source-group split, training config/loss, seed policy,
metric definition, head budget và checkpoint selection trước held-out evaluation.
Báo từng seed, uncertainty, feature dimension, checkpoint/config identity và
phạm vi tái lập. Existing synthetic pilot v1 giữ protocol riêng; tài liệu này
không thay nó hoặc ngầm freeze training recipe mới cho hybrid.

Restart, causality và chunk/step/scan parity là engineering gates trước learning,
không là bằng chứng C đã học event semantics. Các kết luận về natural video,
memory recovery và hiệu năng chỉ được viết sau khi có kết quả tương ứng.

## Table 1. Natural-video representation transfer

**Purpose.** Kiểm external validity: representation học được có chuyển sang task
video tự nhiên hay không, ngoài synthetic core×configuration. Đây là downstream
evidence hỗ trợ, không thay thế kiểm history/configuration ở Table 3 và A1.

**Datasets and metrics.** Something-Something V2 (SSv2) dùng Top-1 accuracy;
EPIC-KITCHENS-100 (EK100) dùng mean top-5 recall (`mT5R`) riêng cho verb, noun và
action trong anticipation setting. Prefix quan sát và anticipation gap phải
giống nhau giữa models và được ghi trong protocol đã freeze.

**Controlled training panel. Frozen encoders with a shared task-head budget.**

| Training regime | Model | SSv2 Top-1 ↑ | EK100 Verb mT5R ↑ | EK100 Noun mT5R ↑ | EK100 Action mT5R ↑ |
|---|---|---:|---:|---:|---:|
| Controlled | Shared encoder + GRU | — | — | — | — |
| Controlled | Shared encoder + causal SSM | — | — | — | — |
| Controlled | Shared encoder + bounded Transformer | — | — | — | — |
| Controlled | Hybrid, unsplit state | — | — | — | — |
| Controlled | MAVT C/D + KV | — | — | — | — |

**External-pretraining panel. Checkpoint and pretraining differences must be declared.**

| Training regime | Model | SSv2 Top-1 ↑ | EK100 Verb mT5R ↑ | EK100 Noun mT5R ↑ | EK100 Action mT5R ↑ |
|---|---|---:|---:|---:|---:|
| External pretraining | VideoMAE | — | — | — | — |
| External pretraining | V-JEPA 2 | — | — | — | — |

Freeze encoders và dùng cùng head budget trong mỗi so sánh phù hợp. External
pretraining khác data/compute phải được giữ ở panel riêng. Nếu một model dùng
bidirectional attention **bên trong prefix đã quan sát**, setting đó được phép
để đo transfer nhưng không được gọi là streaming-causal execution. Không cho
frame sau anticipation cut đi vào encoder hoặc preprocessing.

Protocol/checkpoint references: [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2020-100),
[VideoMAE](https://papers.nips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html),
[V-JEPA 2](https://arxiv.org/abs/2506.09985). Các link định danh benchmark/method,
không cung cấp kết quả cho những ô trống trên.

## Table 2. Image and causal video reconstruction

**Purpose.** Kiểm việc giữ unified tokenizer: cùng hướng mô hình phải tái tạo
ảnh và video, không chỉ thắng event probes. Image evaluation dùng held-out
ImageNet/OpenImages; causal video evaluation dùng held-out OmniWorld. Tách kết
quả theo từng dataset, không gộp PSNR/LPIPS của hai tập ảnh thành một score không
rõ weighting.

Table 2 được typeset thành image và video panels vì readout unit và code bytes
khác nhau. Mỗi image panel lặp lại riêng cho ImageNet và OpenImages. Controlled
rows và external references tiếp tục tách để không che khác biệt training data.

**Image panel — controlled training. Unit: one image.**

| Model | Latent type | Readout KiB/image ↓ | Image PSNR ↑ | Image LPIPS ↓ |
|---|---|---:|---:|---:|
| Original MAVT v3 | Continuous | — | — | — |
| Hybrid, unsplit state | Continuous | — | — | — |
| MAVT C/D + KV | Continuous | — | — | — |

**Image panel — external references, subject to a verified image evaluation path.**

| Model | Latent type | Readout KiB/image ↓ | Image PSNR ↑ | Image LPIPS ↓ |
|---|---|---:|---:|---:|
| Cosmos Tokenizer CV8x8x8 | Continuous | — | — | — |
| MAGVIT-v2 | Discrete | — | — | — |

**Causal video panel — controlled training. Unit: one declared tubelet.**

| Model | Latent type | Readout KiB/tubelet ↓ | Video PSNR ↑ | Video LPIPS ↓ |
|---|---|---:|---:|---:|
| Original MAVT v3 | Continuous | — | — | — |
| Hybrid, unsplit state | Continuous | — | — | — |
| MAVT C/D + KV | Continuous | — | — | — |

**Causal video panel — external references, subject to matched observation support.**

| Model | Latent type | Readout KiB/tubelet ↓ | Video PSNR ↑ | Video LPIPS ↓ |
|---|---|---:|---:|---:|
| Cosmos Tokenizer CV8x8x8 | Continuous | — | — | — |
| MAGVIT-v2 | Discrete | — | — | — |

Tái tạo current tubelet chỉ từ allowed prefix, không future frames. Khóa temporal
unit, resolution, context, dtype, preprocessing và reconstruction queries. Nếu
một checkpoint không hỗ trợ image/tubelet setting tương ứng, không điền score
trước khi xác minh adapter và observation support; tên row không xác nhận sẵn
khả năng đó. Không giữ future context từ một offline encoder rồi gọi cùng
causal-video task.

Báo nhiều code budgets để có rate–distortion comparison; code bytes tính từ
representation thực tế của từng model, không chỉ token count. Với continuous
latent, float bytes **không phải entropy-coded bits per pixel**. Với discrete
latent, index/code representation phải khai báo rõ. C/D snapshot là readout code
tại một thời điểm, không tự là một full-clip codec. Full continuation memory
được báo riêng trong Table 5.

MAGVIT-v2 chỉ có result row sau khi verified checkpoint hoặc reproduction hoàn
thành; không import paper scores vào bảng. External image/video paths và các
latent layouts phải được kiểm theo checkpoint cụ thể. References:
[MAGVIT-v2](https://research.google/pubs/language-model-beats-diffusion-tokenizer-is-key-to-visual-generation/),
[Cosmos Tokenizer inference guide](https://docs.nvidia.com/cosmos/latest/predict1/tokenizer/inference_guide.html).

## Table 3. Event-history retention and configuration modeling under controlled observations

**Purpose.** Đây là bảng trung tâm cho claim về history, configuration và
prediction. Dùng phiên bản mở rộng của synthetic core×configuration, với matched
terminal observations, common suffix dài, continuous configuration factors và
held-out combinations phù hợp. Các extension này cần versioned fixtures/audits
trước run; fixture v1 hiện có không tự cung cấp tất cả task dưới đây.

| Model | History Acc@2 ↑ | History Acc@64 ↑ | Composition OOD Acc ↑ | Config NRMSE ↓ | Future Rollout L1 ↓ |
|---|---:|---:|---:|---:|---:|
| Last tubelet | — | — | — | — | — |
| Orderless temporal mean | — | — | — | — | — |
| Shared encoder + GRU | — | — | — | — | — |
| Shared encoder + causal SSM | — | — | — | — | — |
| Shared encoder + bounded Transformer | — | — | — | — | — |
| Hybrid, unsplit state | — | — | — | — | — |
| MAVT C/D + KV | — | — | — | — | — |

`History Acc@k` đo history-label accuracy sau **k accepted observation steps của
common suffix**; k không phải số raw frames hoặc seconds. Cùng lịch observation
và tubelets giữa models. Với reversal diagnostic, report direction-pair scores
và aggregation rule, không thay bằng four-way score che motion-family shortcut.

`Composition OOD Acc` dùng combinations chưa thấy trong representation training;
khóa factor crossing và source split trước run. Không gọi held-out texture/camera
sources của fixture v1 là composition OOD nếu combinations vẫn giống train.
`Config NRMSE` dùng normalization fit trên train data, cùng factor definitions
giữa models; report từng factor ở appendix. Khóa công thức denominator và cách
aggregate trước held-out. Compact/wide classification v1 không tự trở thành
continuous-configuration regression.

`Future Rollout L1` dùng **actual model transition và decoder**, không dùng một
predictor riêng đủ mạnh để che chất lượng state. Causal cue cần có trong observed
prefix để future task có thể được xác định. Hai clips có RGB hiện tại giống nhau
không có nghĩa physical state giống nhau, nên fixture phải chỉ rõ history cue
và future target. Với controls không có transition/decoder tự nhiên, chỉ chấm
khi đã có một baseline rule cùng task được freeze; chưa có thì ô vẫn là —.

Dùng common readout/head budget cho representation comparisons; không bắt GRU,
SSM hoặc Transformer đổi state thành C/D để tham gia. Diễn giải C/D riêng ở A1,
không lẫn model-level representation quality với internal factorization claim.

## Table 4. Recovery from missing observations using historical evidence

**Purpose.** Kiểm lịch sử có giúp đọc vùng thiếu hay không. Bắt đầu bằng synthetic
data có exposure ground truth; sau đó chuyển sang những OmniWorld masking tasks
có evidence/targets thích hợp. Natural-video pixels che nhân tạo không mặc định
có amodal hoặc 3D-surface visibility ground truth.

| Model | Seen-before L1@8 ↓ | Seen-before L1@64 ↓ | Never-seen L1 ↓ | Full-gap Rollout L1 ↓ | History Gain ↑ |
|---|---:|---:|---:|---:|---:|
| Copy last valid | — | — | — | — | — |
| Prefix-only flow warp | — | — | — | — | — |
| Shared encoder + GRU | — | — | — | — | — |
| Shared encoder + bounded Transformer | — | — | — | — | — |
| Hybrid, unsplit state | — | — | — | — | — |
| MAVT C/D + KV | — | — | — | — | — |

Seen-before lag dùng cùng accepted-step convention, tính từ exposure được khai
báo tới readout. `Full-gap Rollout L1` đo sau khoảng không observation với cùng
lịch bước và missing policy. Copy-last-valid baseline dùng observation thực sự
khả dụng; flow warp chỉ được ước lượng bằng prefix. Không dùng GT flow hoặc
future frame để giúp một baseline dưới tên “prefix-only”.

\[
\mathrm{History\ Gain}=L_{\mathrm{no\ history}}-L_{\mathrm{correct\ history}}.
\]

Hai điều kiện giữ **cùng current observation, mask và query**; chỉ đổi history.
Khóa cách loại history, lag, availability và metric normalization trước đo.
Wrong-history control đưa ở appendix để phân biệt truy hồi evidence đúng với
chỉ hưởng lợi từ thêm input bất kỳ.

Chỉ gắn nhãn `Never-seen` khi exposure GT xác nhận chưa có evidence tương ứng.
Một fixed image rectangle bị mask suốt clip không đủ nếu camera/object motion
làm cùng surface xuất hiện ở vị trí khác. Báo vùng seen-before, other-view-only
và never-seen riêng; không gọi hallucination ở never-seen là memory recovery.

Cross-view recovery là panel riêng, chỉ thêm sau khi có verified multi-view
adapters, camera conventions và visibility/exposure GT. Natural occlusion và
amodal readout khác pixel masking: occluder RGB vẫn hợp lệ, còn hidden-object
query/targets cần được định nghĩa riêng.

## Table 5. Quality–memory–latency trade-offs in streaming inference

**Purpose.** Đo chất lượng cùng chi phí thực khi stream dài. Tách comparisons
theo **same full memory** và **same training compute**, không gọi cùng số C/D
values là cùng memory khi một model có thêm KV.

| Model | Params M ↓ | Readout KiB ↓ | Restart KiB ↓ | Peak MiB ↓ | Step Latency ms ↓ | History Acc@64 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Shared encoder + GRU | — | — | — | — | — | — |
| Shared encoder + causal SSM | — | — | — | — | — | — |
| Shared encoder + bounded Transformer | — | — | — | — | — | — |
| Hybrid, unsplit state | — | — | — | — | — | — |
| MAVT C/D, trained without KV | — | — | — | — | — | — |
| MAVT C/D + KV | — | — | — | — | — | — |

`Readout KiB` đếm code decoder cần ở một readout unit; `Restart KiB` đếm toàn bộ
persistent memory để tiếp tục chính xác, gồm C/D, KV và metadata. Với model khác,
mọi recurrent/conv/position cache cần cho restart đều phải tính. Serialization
container overhead và shared model parameters phải được giải thích riêng.

`Peak MiB` dùng actual inference allocation với cùng reporting convention;
không chỉ cộng tensors vào logical state rồi gọi là peak memory. Khai báo hardware
class, precision, batch size, view count, resolution, query count, context/sequence
length, warmup và synchronization. Step latency phải xác định có gồm observation
encoder, cache update, recurrence và decoder hay không; main comparison dùng
cùng phạm vi end-to-end. Không chọn kernel-only time cho một model và whole-model
time cho model khác.

Throughput/chunk-size/scan scaling đưa thành figure hoặc appendix. Python reference
và optimized kernels cần phân biệt; associative algebra không tự chứng minh
throughput tốt hơn GRU. Trained-without-KV là một independent training ablation,
không phải lấy trained-with-KV checkpoint rồi flush cache lúc evaluation.

## Table A1. Information accessible from C and D

**Purpose.** Kiểm riêng phân vai C/D, hỗ trợ claim trung tâm. Nếu paper thiếu
không gian, ưu tiên bảng này hơn một auxiliary downstream table. Đây là probe
readability, chưa chứng minh decoder hoặc transition thực sự sử dụng thông tin.

| Frozen Representation | History Acc@64 ↑ | Event OOD Acc ↑ | Config NRMSE ↓ |
|---|---:|---:|---:|
| Untrained C | — | — | — |
| Untrained D | — | — | — |
| Trained C1 + C2 | — | — | — |
| Trained D1 + D2 | — | — | — |
| Trained C + D | — | — | — |

C/D untrained dùng matched initialization và cùng extraction support. Mọi row
dùng cùng probe family và train-only fit/normalization; report feature dimension
và head budget, vì C+D có thể nhiều dimensions hơn từng bank. `Event OOD` phải
tham chiếu đúng factor split đã freeze của Table 3, không đổi nghĩa OOD giữa bảng.

Cross scores C→configuration và D→history/event không bắt buộc về chance. Điều
cần kiểm là C có giữ diễn biến và D có giữ configuration, cùng mức information
overlap, thay vì định nghĩa disentanglement bằng việc một bank hoàn toàn không
có thông tin của bank kia. So trained với untrained trước khi claim đã học.

## Table A2. Architectural, objective, and masking ablations

**Purpose.** Tách contribution của architecture, learning signals và masking.
Mỗi row là independent training run với cùng comparison policy; không thay loss
hoặc architecture trên checkpoint đã train rồi gọi là training ablation.

| Panel | Configuration | History Acc@64 ↑ | Config NRMSE ↓ | Missing L1 ↓ |
|---|---|---:|---:|---:|
| Reference | Full MAVT C/D + KV | — | — | — |
| Architecture | Unsplit state, matched readout budget | — | — | — |
| Architecture | One temporal stage | — | — | — |
| Architecture | Trained without KV | — | — | — |
| Architecture | Direct intra-cell D→C disabled | — | — | — |
| Objective | C-history loss weight = 0 | — | — | — |
| Objective | Future loss weight = 0 | — | — | — |
| Masking | Isotropic masks, matched masked volume | — | — | — |

Final hybrid loss phải được freeze trước khi diễn giải các objective rows; hiện
chưa có ngầm định về weights ngoài việc ablation đặt đúng term đã định nghĩa
về zero. Mỗi panel khai báo parameter/state/compute changes khi không thể giữ
mọi budget bằng nhau. Masking comparison match masked volume và observation
support phù hợp, không để isotropic/anisotropic khác nhau chỉ vì tỷ lệ missing.

Direct intra-cell D→C cut chỉ loại `a_CD` trong coupled cells. Đường
`D1 → mixer → GQA2 → C2` vẫn tồn tại; không gọi đó là cắt mọi D→C feedback.
One-stage/unsplit/cache-off variants cần implementation và correctness gates
riêng. Trained cache-off không đồng nghĩa one-time flush trong A3.

## Table A3. State interventions at a fixed history cut

**Purpose.** Phân tích nơi lưu và khả năng tái nạp event information trong một
model đã train. Cố định cut ngay sau history cue, giữ cùng timestamps, current
observations và continuation schedule giữa các rows.

| Intervention | C/D at Cut | KV at Cut | History Acc@64 ↑ | Future Rollout L1 ↓ |
|---|---|---|---:|---:|
| Intact state | Keep | Keep | — | — |
| Flush KV once | Keep | Clear | — | — |
| Reset C/D | Clear | Keep | — | — |
| Reset both | Clear | Clear | — | — |
| Disable KV throughout continuation | Keep | Disabled | — | — |

Clear/reset chỉ thay các memory contents đã nêu, giữ clock/ring metadata hợp lệ
và cùng physical schedule. `Flush KV once` cho phép cache được ghi lại sau cut;
`Disable KV throughout continuation` chặn cache trong cả continuation. Row sau
là distribution-shift diagnostic, không phải trained-without-KV model ở T5/A2.

KV có thể tái nạp history information vào C/D, đặc biệt khi KV2 mang lower-state
readout. Vì thế reset C/D/keep KV không bắt buộc về chance, và lag lớn hơn window
không tự chứng minh mọi thông tin chỉ sống trong C/D. Báo intact và intervention
scores cùng uncertainty, không chọn cut sau khi thấy kết quả.

Flush KV **ngay trước decode** với cùng C/D/query phải giữ RGB bất biến. Đây là
technical check của C/D-only decoder; không trộn nó với learned-memory evidence
hoặc xem như một training ablation.

## Freeze checklist before execution

Trước khi chạy bộ bảng này, cần versioned datasets/adapters cho từng task,
reproducible external checkpoints, training/head budgets, metric formulas và
normalization, cut/lag/mask/exposure definitions, seeds/uncertainty và checkpoint
selection. Các nội dung chưa khóa được hoàn thành trong
[hybrid protocol](mavt-hybrid-v2-protocol.md), với config/version mới khi cần.

Không suy kết quả từ việc tài liệu hoặc table skeleton đã có trong repo. Chỉ
điền ô sau khi run tương ứng hoàn tất, protocol đã được freeze và provenance
được kiểm; giữ **—** cho mọi điều kiện chưa đo.
