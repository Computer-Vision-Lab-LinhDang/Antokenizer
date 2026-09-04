# Vì sao nhánh content dùng slot-attention thất bại

**Ngày:** 2026-09-04 · **Đo trên:** `runs/deepseek/s3@15k`, `runs/rd10k@{3000,10000}` · **Script:** `scripts/split_health.py`, `split_axis.py`, `why_stuck.py`, `salvage.py`, `dead_gradient.py`, `ablate_branches.py`

## Kết luận

Phép tách content/detail của MAVT không hoạt động, và nguyên nhân **không phải** thiếu áp lực huấn luyện. Đường truyền gradient tới phép gán patch→slot bị ngắt về mặt số học ngay từ đầu, do softmax bão hoà bởi biên độ logit khổng lồ. Mọi cách sửa ở tầng loss đều vô ích vì chúng tác động vào một đường đã đứt.

Bằng chứng dứt điểm: `w_content = 1.0` (ngang trọng số L1) chạy 10.000 step làm `content_recon_error` đứng nguyên ở **1.000**, tức đúng mức của lời giải "chỉ đoán trung bình".

## Chuỗi nhân quả

**1. Feature backbone có massive activations.** 99,9% năng lượng nằm ở một vector trung bình chung cho mọi patch; phần dao động mang toàn bộ thông tin chỉ chiếm 0,1%. Đây là hiện tượng đã được ghi nhận rộng rãi trong transformer ([Massive Activations in LLMs, 2024](https://arxiv.org/html/2402.17762v1); [artifact/register tokens trong ViT](https://papers.nips.cc/paper_files/paper/2025/file/51bae6441380c4c4cab87f430be89ac2-Paper-Conference.pdf)): vài chiều có biên độ cực lớn đóng vai trò bias hằng số.

**2. Logit bị thành phần chung chi phối.** `logit[c,n] = C[c]·x[n]/√D`. Vì `x[n] ≈ mean + dev` với `‖mean‖ >> ‖dev‖`, số hạng `C[c]·mean` quyết định thứ hạng giữa các slot, và nó **giống nhau ở mọi patch**. Đo được: `|logit trung bình| / |logit|` = 1,0000.

**3. Softmax bão hoà thành one-hot hằng số.** `|logit|` trung bình 109,8 → entropy trên trục slot = **0,0000** (tối đa là ln 64 = 4,159). Cùng một slot thắng ở mọi patch nên `x_approx` là hằng số: biên độ dao động của nó **0,000** so với 221,9 của feature thật. Độ phân tán trọng số attention theo patch = 0,0000.

**4. Gradient chết.** `‖∂L/∂logit‖ = 5,0×10⁻¹⁰`. Chia logit cho 100 làm entropy hồi phục về 3,98 và gradient tăng **73.000 lần** lên 3,7×10⁻⁵. Đây là mắt xích quyết định: phép gán không thể học, bất kể trọng số loss.

**5. Slot trôi về gần ngẫu nhiên.** Không có gradient hữu ích, span của 64 slot chỉ bắt **6,3%** phương sai (s3@15k) — một không gian con 64 chiều **ngẫu nhiên** trong 1152 chiều đã bắt 5,6%. Sau 10k step có loss ép, lên được 32%. Đáng chú ý: slot vẫn đa dạng (cos giữa các slot 0,082) và đủ hạng 64/64 — **đa dạng không đồng nghĩa với mang thông tin**.

**6. Kiến trúc rỗng.** Content vô dụng → detail gánh toàn bộ → ở window 2 bị gộp 4:1 → thua SD-VAE 3,5 dB tại cùng 4.096 float.

## Bằng chứng

Mọi cách đọc lại 64 slot đã train (`content_err`: 1,0 = vô dụng, 0 = hoàn hảo, chuẩn hoá theo phương sai quanh trung bình):

| Cách đọc | content_err |
|---|---|
| softmax hiện tại | 1,001 |
| cosine + nhiệt độ 0,1 / 0,05 / 0,01 | 1,017 / 1,013 / 1,013 |
| LayerNorm cả hai rồi dot | 1,009 |
| bình phương tối thiểu trên span(C) — trần của chính bộ slot | 0,682 |
| 64 hướng PCA tốt nhất — trần tuyệt đối | **0,000** |

Hai điều quan trọng: sửa cách đọc **sau khi đã train** không cứu được (slot đã hỏng); và thông tin thì **hoàn toàn nén được** vào 64 chiều, nên đây là lỗi cơ chế chứ không phải bài toán bất khả.

Ablation đóng góp vào PSNR: content **0,12–1,33 dB** tuỳ checkpoint, detail 15–22 dB.

## Sai lầm thiết kế gốc

Slot-attention giả định phép gán là học được, điều chỉ đúng khi logit ở thang O(1). Attention chuẩn giữ được điều đó vì Q, K là phép chiếu học được có weight decay. Ở đây `C` và `x` là **feature thô của backbone**, biên độ tăng dần trong lúc train, trong khi nhiệt độ vẫn cố định ở `1/√D` — hằng số hiệu chỉnh cho đầu vào phương sai đơn vị. Nói ngắn: **nhiệt độ sai hiệu chỉnh so với massive activations, gây bão hoà không thể đảo ngược trước khi việc học kịp bắt đầu**.

## Vì sao không phát hiện sớm

- `slot_diversity` luôn trông khoẻ mạnh, vì slot thực sự đa dạng. Chỉ số này đo sai thứ cần đo.
- `cd_residual_ratio` được log ở mức ~1,0 suốt nhiều run, đúng nghĩa "phép tách không làm gì", nhưng không ai đọc nó như một cảnh báo.
- Tái tạo vẫn chạy được nhờ nhánh detail, nên không có lỗi ồn ào nào.

## Hướng đi

1. **Nút cổ chai tuyến tính thay slot-attention.** Không softmax thì không bão hoà; PCA cho thấy 64 chiều là đủ (lỗi 0,000). Đây là lựa chọn có bằng chứng mạnh nhất.
2. **Nếu giữ attention:** chuẩn hoá query/key (LayerNorm hoặc L2) cộng nhiệt độ học được, và phải áp dụng **từ đầu quá trình train**. Thay ở khâu đọc sau khi đã train thì không có tác dụng (đã đo).
3. **Xử lý từ thượng nguồn:** register token để hút massive activations ra khỏi các patch token.

## Việc KHÔNG nên thử lại

Chỉnh trọng số loss content, đổi nhiệt độ softmax, hay chuẩn hoá logit ở khâu đọc sau khi train. Cả ba đã đo, đều cho ~1,0.
