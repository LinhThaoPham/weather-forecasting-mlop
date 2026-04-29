# BÁO CÁO ĐỒ ÁN: HỆ THỐNG MLOPS DỰ BÁO THỜI TIẾT VIỆT NAM

**Môn học:** Data Engineering
**Công nghệ:** FastAPI · Prophet · LSTM · Docker · Microservices

---

# MỤC LỤC

- Chương 1: Giới thiệu đề tài
- Chương 2: Cơ sở lý thuyết
- Chương 3: Kiến trúc hệ thống
- Chương 4: Data Pipeline
- Chương 5: Mô hình Machine Learning
- Chương 6: API Services (Microservices)
- Chương 7: Giao diện Dashboard
- Chương 8: Docker và Containerization
- Chương 9: Kết quả và đánh giá
- Chương 10: Kết luận và hướng phát triển

---

# CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI

## 1.1 Đặt vấn đề

Dự báo thời tiết là một bài toán quan trọng trong đời sống và sản xuất. Các hệ thống dự báo truyền thống thường dựa trên mô hình vật lý phức tạp, đòi hỏi tài nguyên tính toán lớn và khó tùy biến cho từng khu vực cụ thể. Trong bối cảnh đó, việc ứng dụng Machine Learning vào dự báo thời tiết mở ra hướng tiếp cận mới — cho phép xây dựng các mô hình nhẹ, dễ triển khai và có khả năng học từ dữ liệu lịch sử của từng địa phương.

Tuy nhiên, một thách thức lớn trong thực tế là: **xây dựng mô hình ML chỉ chiếm khoảng 20% công việc, còn 80% nằm ở việc vận hành, triển khai và bảo trì hệ thống** (theo nghiên cứu của Google, 2015). Đây chính là lý do MLOps (Machine Learning Operations) ra đời — kết hợp các thực hành DevOps vào quy trình ML để tự động hóa toàn bộ vòng đời từ thu thập dữ liệu, training, serving đến monitoring.

## 1.2 Mục tiêu đồ án

Xây dựng một hệ thống MLOps hoàn chỉnh cho bài toán dự báo thời tiết tại Việt Nam, bao gồm:

1. **Data Pipeline tự động**: Thu thập dữ liệu thời tiết real-time từ Open-Meteo API cho 6 thành phố Việt Nam.
2. **Hybrid ML Model**: Kết hợp hai mô hình Prophet (seasonal patterns) và LSTM (short-term temporal patterns) theo chiến lược Ensemble Weighted Averaging.
3. **Microservices Architecture**: Triển khai hệ thống dưới dạng 5 microservices độc lập giao tiếp qua REST API.
4. **Containerization**: Đóng gói toàn bộ hệ thống bằng Docker và Docker Compose.
5. **Dashboard tương tác**: Giao diện web hiển thị kết quả dự báo theo giờ và theo ngày.

## 1.3 Phạm vi đồ án

- **Dữ liệu**: Nhiệt độ, độ ẩm, mây che phủ — lấy từ Open-Meteo (miễn phí).
- **Thành phố**: 6 thành phố lớn tại Việt Nam (Hà Nội, TP.HCM, Đà Nẵng, Hải Phòng, Nha Trang, Đà Lạt).
- **Dự báo**: Theo giờ (72 giờ tới) và theo ngày (trung bình 3 ngày tới).
- **Triển khai**: Docker Compose trên máy cục bộ.

## 1.4 Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|---|---|---|
| Ngôn ngữ | Python | 3.10 |
| API Framework | FastAPI | 0.104+ |
| ML - Seasonal | Facebook Prophet | 1.1.5 |
| ML - Deep Learning | TensorFlow / Keras | 2.14+ |
| Data Processing | Pandas, NumPy, Scikit-learn | -- |
| Frontend | HTML, JavaScript, Chart.js, Tailwind CSS | -- |
| Container | Docker, Docker Compose | -- |
| Data Source | Open-Meteo API | Free tier |

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 MLOps (Machine Learning Operations)

### 2.1.1 Định nghĩa

MLOps là một tập hợp các thực hành kết hợp Machine Learning, DevOps và Data Engineering nhằm triển khai và duy trì các hệ thống ML trong sản xuất một cách đáng tin cậy và hiệu quả.

### 2.1.2 Vòng đời MLOps

```
Thu thập dữ liệu --> Tiền xử lý --> Feature Engineering --> Training
        ^                                                      |
    Monitoring  <--  Serving  <--  Deployment  <--  Đánh giá mô hình
```

| Giai đoạn | Công cụ trong đồ án |
|---|---|
| Thu thập dữ liệu | fetch_data.py — gọi Open-Meteo API |
| Tiền xử lý | preprocess.py — làm sạch, chuẩn hóa |
| Feature Engineering | feature_engineering.py — sliding window, temporal features |
| Training | train_hourly.py, train_daily.py, train_lstm_colab.py |
| Model Serving | FastAPI microservices (4 services) |
| Deployment | Docker Compose |
| Dashboard | Vanilla JS + Streamlit |

### 2.1.3 Microservices vs Monolith

| Đặc điểm | Monolith | Microservices (Đồ án này) |
|---|---|---|
| Triển khai | Deploy cả hệ thống | Deploy từng service riêng |
| Scaling | Scale toàn bộ | Scale service cần thiết |
| Fault isolation | 1 lỗi sập hệ thống | 1 service lỗi, các service khác vẫn chạy |
| Phức tạp | Thấp | Cao hơn (cần orchestration) |


## 2.2 Machine Learning — Nền tảng

### 2.2.1 Định nghĩa Machine Learning

Machine Learning (ML) là một nhánh của Trí tuệ Nhân tạo (AI), cho phép máy tính **tự động học từ dữ liệu** mà không cần lập trình tường minh từng quy tắc. Thay vì viết `if nhiệt_độ > 35: print("nóng")`, ML sẽ tự phát hiện ngưỡng này từ hàng nghìn mẫu dữ liệu.

**Phân loại ML:**

| Loại | Mô tả | Ví dụ trong đồ án |
|---|---|---|
| Supervised Learning | Học có giám sát — có input + label | Dự báo nhiệt độ (input: dữ liệu quá khứ, label: nhiệt độ tương lai) |
| Unsupervised Learning | Học không giám sát — chỉ có input | Không sử dụng |
| Reinforcement Learning | Học tăng cường — reward/punishment | Không sử dụng |

Đồ án sử dụng **Supervised Learning** dạng **Regression** (dự đoán giá trị liên tục — nhiệt độ °C).

### 2.2.2 Supervised Learning cho Time Series

Trong bài toán dự báo thời tiết, supervised learning được áp dụng như sau:

```
Input (X):   Dữ liệu quá khứ [t-24, t-23, ..., t-1]
Output (y):  Giá trị tương lai [t, t+1, ..., t+71]

Mục tiêu: Tìm hàm f sao cho f(X) ≈ y
```

**Train/Test Split theo thời gian:**

```
|←—— Training Data (80%) ——→|←— Test Data (20%) —→|
|  2024-01  ............  2025-09  |  2025-09 ... 2026-04  |
```

Lưu ý: Không dùng random split như classification, vì dữ liệu thời gian có tính tuần tự.

### 2.2.3 Hàm mất mát (Loss Function)

Hàm mất mát đo lường **sai số giữa giá trị dự đoán và giá trị thực**:

**MSE (Mean Squared Error)** — Loss chính của LSTM:

```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

- Phạt nặng các sai số lớn (vì bình phương)
- Luôn ≥ 0, càng nhỏ càng tốt
- Đơn vị: °C² (khó giải thích trực quan)

**MAE (Mean Absolute Error)** — Metric phụ:

```
MAE = (1/n) × Σ|y_actual - y_predicted|
```

- Dễ hiểu hơn MSE: "trung bình sai lệch bao nhiêu °C"
- Không phạt nặng outlier như MSE

**Trong đồ án:**
```python
model.compile(
    loss='mse',       # Tối ưu theo MSE
    metrics=['mae']   # Theo dõi thêm MAE
)
```

### 2.2.4 Overfitting và Underfitting

| Trạng thái | Biểu hiện | Nguyên nhân |
|---|---|---|
| Underfitting | Train loss cao, Val loss cao | Model quá đơn giản, ít epochs |
| Good fit | Train loss thấp, Val loss thấp | Cân bằng |
| Overfitting | Train loss rất thấp, Val loss cao | Model quá phức tạp, quá nhiều epochs |

**Kỹ thuật chống Overfitting trong đồ án:**

1. **Dropout (0.2)**: Tắt ngẫu nhiên 20% neurons trong mỗi lần train → buộc model không phụ thuộc vào một neuron cụ thể.
2. **Early Stopping (patience=10)**: Dừng train khi val_loss không giảm sau 10 epochs liên tiếp.
3. **ReduceLROnPlateau**: Giảm learning rate ×0.5 khi val_loss không cải thiện → giúp hội tụ mịn hơn.

### 2.2.5 Chuẩn hóa dữ liệu (Normalization)

Neural network hoạt động tốt nhất khi dữ liệu nằm trong khoảng [0, 1] hoặc [-1, 1]:

**MinMax Scaler:**
```
x_normalized = (x - x_min) / (x_max - x_min)
```

Ví dụ: Nhiệt độ Hà Nội dao động 5°C — 42°C:
```
Nhiệt độ 28°C → (28 - 5) / (42 - 5) = 0.62
```

**Tại sao cần normalize?**
- Gradient descent hội tụ nhanh hơn
- Tránh neuron bão hòa (saturation) trong activation functions
- Các features có scale khác nhau được xử lý công bằng

---

## 2.3 Deep Learning — Nền tảng

### 2.3.1 Neural Network cơ bản

Neural Network (mạng nơ-ron nhân tạo) mô phỏng cách hoạt động của não người, gồm các lớp neurons kết nối với nhau:

```
Input Layer        Hidden Layer(s)       Output Layer
[x1] ──┐
       ├──→ [h1] ──┐
[x2] ──┤           ├──→ [y]
       ├──→ [h2] ──┘
[x3] ──┘
```

Mỗi kết nối có một **trọng số (weight)** và mỗi neuron có **bias**:

```
output = activation(w1*x1 + w2*x2 + w3*x3 + bias)
```

### 2.3.2 Activation Functions

Activation function quyết định neuron có "bật" hay không:

| Hàm | Công thức | Đặc điểm | Dùng trong đồ án |
|---|---|---|---|
| ReLU | max(0, x) | Nhanh, đơn giản, tránh vanishing gradient | ✅ LSTM layers |
| Sigmoid | 1/(1+e^(-x)) | Output [0,1], dùng cho gate | ✅ Bên trong LSTM gate |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | Output [-1,1] | ✅ Bên trong LSTM cell |
| Linear | x | Không biến đổi | ✅ Output layer (regression) |

### 2.3.3 Backpropagation và Gradient Descent

**Quá trình training:**

```
1. Forward Pass: Tính output từ input → qua các layers → ra prediction
2. Loss Calculation: So sánh prediction với ground truth → tính MSE
3. Backward Pass: Tính gradient (đạo hàm) của loss theo từng weight
4. Weight Update: w_new = w_old - learning_rate × gradient
5. Lặp lại từ bước 1 cho batch tiếp theo
```

**Adam Optimizer** (sử dụng trong đồ án):

Adam (Adaptive Moment Estimation) kết hợp ưu điểm của 2 optimizer:
- **Momentum**: Tích lũy gradient quá khứ → vượt qua local minima
- **RMSprop**: Tự điều chỉnh learning rate cho từng parameter

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
```

Tại sao chọn Adam?
- Tự động điều chỉnh learning rate
- Hội tụ nhanh hơn SGD trên hầu hết bài toán
- Ít nhạy cảm với hyperparameter

### 2.3.4 Batch và Epoch

| Khái niệm | Định nghĩa | Trong đồ án |
|---|---|---|
| Sample | 1 cặp (input, output) | 1 cửa sổ 24h → 72h |
| Batch | Nhóm samples xử lý cùng lúc | 32 samples |
| Epoch | 1 lần duyệt hết toàn bộ data | Tối đa 100 epochs |
| Iteration | 1 lần update weights | n_samples / batch_size |

Ví dụ: Với 17,000 samples, batch_size=32:
- Mỗi epoch = 17,000 / 32 ≈ 531 iterations
- Tối đa 100 epochs = 53,100 iterations

---

## 2.4 Recurrent Neural Network (RNN)

### 2.4.1 Tại sao cần RNN?

Neural Network thông thường (feedforward) xử lý mỗi input **độc lập** — không nhớ các input trước đó. Nhưng dữ liệu thời tiết có **tính tuần tự**: nhiệt độ lúc 14h phụ thuộc vào nhiệt độ lúc 13h, 12h,...

RNN giải quyết vấn đề này bằng cách thêm **vòng lặp (recurrence)**:

```
        ┌──────────┐
        │          │
x[t] ──→│  RNN     │──→ h[t] (output)
        │  Cell    │
h[t-1]─→│          │──→ h[t] (truyền sang bước tiếp)
        └──────────┘
```

Mỗi bước thời gian, RNN nhận 2 input:
- `x[t]`: Dữ liệu tại thời điểm t
- `h[t-1]`: "Bộ nhớ" từ bước trước (hidden state)

### 2.4.2 Vấn đề Vanishing Gradient

RNN truyền thống gặp vấn đề **vanishing gradient**: khi chuỗi dài (ví dụ 24 giờ), gradient bị nhân nhiều lần qua các bước → trở nên cực nhỏ → model không học được dependency dài hạn.

```
Gradient tại bước 1:
g = g₂₄ × g₂₃ × g₂₂ × ... × g₁

Nếu mỗi gᵢ < 1 → g → 0 (vanishing)
Nếu mỗi gᵢ > 1 → g → ∞ (exploding)
```

**LSTM được thiết kế để giải quyết chính xác vấn đề này.**

---

## 2.5 LSTM (Long Short-Term Memory)

### 2.5.1 Kiến trúc LSTM Cell

LSTM thêm một **Cell State** (bộ nhớ dài hạn) chạy xuyên suốt chuỗi, được điều khiển bởi 3 cổng (gate):

```
                    ┌─────────────────────────────────┐
                    │         Cell State (Cₜ)          │
         ×─────────┤  ←forget→  ←+input→  ──────────→├──→ Cₜ
         │         └─────────────────────────────────┘
         │                        ↑
    ┌────┴────┐            ┌──────┴──────┐      ┌──────────┐
    │ Forget  │            │   Input     │      │  Output  │
    │  Gate   │            │   Gate      │      │   Gate   │
    │ σ(Wf)   │            │ σ(Wi)×tanh  │      │  σ(Wo)   │
    └────┬────┘            └──────┬──────┘      └────┬─────┘
         │                        │                   │
         └────────┬───────────────┘                   │
                  │                                   │
            [hₜ₋₁, xₜ]                          hₜ = Oₜ × tanh(Cₜ)
```

**3 cổng chi tiết:**

1. **Forget Gate (Cổng quên):** Quyết định bao nhiêu % thông tin cũ cần giữ lại.
   ```
   fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)    // output ∈ [0, 1]
   // 0 = quên hết, 1 = giữ nguyên
   ```

2. **Input Gate (Cổng nhập):** Quyết định thông tin mới nào cần ghi nhớ.
   ```
   iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     // bao nhiêu % cần ghi
   C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  // thông tin mới
   ```

3. **Output Gate (Cổng xuất):** Quyết định output từ cell state.
   ```
   oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
   hₜ = oₜ × tanh(Cₜ)
   ```

**Cập nhật Cell State:**
```
Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ
     (giữ lại)    (thêm mới)
```

### 2.5.2 Tại sao LSTM giải quyết Vanishing Gradient?

Cell State `Cₜ` truyền thông tin qua phép **cộng** (additive) thay vì phép **nhân** (multiplicative) như RNN:

```
RNN:  hₜ = tanh(W × hₜ₋₁ + U × xₜ)         ← nhân liên tục → vanishing
LSTM: Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ             ← cộng → gradient ổn định
```

Forget gate `fₜ` học cách giữ gradient ≈ 1 cho các dependency dài hạn.

### 2.5.3 Kiến trúc Encoder-Decoder trong đồ án

Đồ án sử dụng kiến trúc **Encoder-Decoder** cho bài toán sequence-to-sequence (nhiều input → nhiều output):

```
Input: [t-24, t-23, ..., t-1]    ← 24 giá trị nhiệt độ gần nhất
              ↓
    ┌─────────────────────┐
    │  LSTM Encoder (128) │    ← Nén lịch sử thành context vector
    │  Dropout (0.2)      │
    │  Dense (64)         │
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │  RepeatVector (72)  │    ← Lặp context cho mỗi time step output
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │  LSTM Decoder (64)  │    ← Sinh chuỗi dự báo tuần tự
    │  Dropout (0.2)      │
    │  TimeDistributed    │
    │  Dense (1)          │
    └─────────┬───────────┘
              ↓
Output: [t+1, t+2, ..., t+72]   ← 72 giá trị dự báo
```

**Giải thích từng layer:**

| Layer | Input Shape | Output Shape | Vai trò |
|---|---|---|---|
| Input | (batch, 24, 1) | (batch, 24, 1) | Nhận 24 giá trị nhiệt độ |
| LSTM Encoder (128) | (batch, 24, 1) | (batch, 128) | Nén 24 bước → 128-dim vector |
| Dropout (0.2) | (batch, 128) | (batch, 128) | Chống overfitting |
| Dense (64) | (batch, 128) | (batch, 64) | Giảm chiều context vector |
| RepeatVector (72) | (batch, 64) | (batch, 72, 64) | Copy context 72 lần |
| LSTM Decoder (64) | (batch, 72, 64) | (batch, 72, 64) | Tạo 72 output tuần tự |
| Dropout (0.2) | (batch, 72, 64) | (batch, 72, 64) | Chống overfitting |
| TimeDistributed Dense (1) | (batch, 72, 64) | (batch, 72, 1) | Mỗi step → 1 giá trị nhiệt độ |

### 2.5.4 Hyperparameters chi tiết

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| Lookback window | 24 giờ | Đủ 1 chu kỳ ngày-đêm để bắt daily pattern |
| Forecast horizon | 72 giờ | 3 ngày — đủ dài để hữu ích, đủ ngắn để chính xác |
| Encoder units | 128 | Đủ capacity để encode 24-step sequence |
| Decoder units | 64 | Nhỏ hơn encoder vì decode đơn giản hơn |
| Dropout rate | 0.2 | Chuẩn cho time series — không quá mạnh |
| Learning rate | 0.001 | Default tốt cho Adam optimizer |
| Batch size | 32 | Trade-off tốt giữa tốc độ và ổn định gradient |
| Max epochs | 100 | Kết hợp EarlyStopping, thường dừng ở epoch 30-50 |
| Early stopping patience | 10 | Đủ kiên nhẫn để tránh dừng sớm do fluctuation |

---

## 2.6 Chuỗi thời gian (Time Series)

### 2.6.1 Định nghĩa

Chuỗi thời gian là một dãy các điểm dữ liệu được đo theo thứ tự thời gian, thường ở các khoảng cách đều nhau.

### 2.6.2 Các thành phần phân rã

Một chuỗi thời gian Y(t) có thể phân rã thành:

```
Y(t) = Trend(t) + Seasonality(t) + Residual(t)    (Additive)
Y(t) = Trend(t) × Seasonality(t) × Residual(t)    (Multiplicative)
```

| Thành phần | Mô tả | Ví dụ thời tiết Hà Nội |
|---|---|---|
| Trend | Xu hướng dài hạn | Nhiệt độ tăng nhẹ do biến đổi khí hậu |
| Yearly Seasonality | Chu kỳ theo năm | Mùa hè (6-8) nóng 35°C, mùa đông (12-2) lạnh 15°C |
| Weekly Seasonality | Chu kỳ theo tuần | Nhẹ — hiệu ứng đảo nhiệt đô thị cuối tuần |
| Daily Seasonality | Chu kỳ theo ngày | Sáng 22°C → trưa 32°C → tối 25°C |
| Residual | Nhiễu ngẫu nhiên | Mưa giông bất chợt, gió mùa |

### 2.6.3 Autocorrelation

Autocorrelation đo mức tương quan giữa một giá trị với chính nó tại các bước thời gian trước:

```
ACF(lag=1):  Nhiệt độ giờ này ↔ giờ trước      → rất cao (~0.98)
ACF(lag=24): Nhiệt độ giờ này ↔ cùng giờ hôm qua → cao (~0.85)
ACF(lag=168): Nhiệt độ giờ này ↔ cùng giờ tuần trước → khá (~0.70)
```

LSTM tận dụng autocorrelation thông qua lookback window = 24 (1 ngày).

---

## 2.7 Facebook Prophet

### 2.7.1 Mô hình toán học

Prophet dựa trên mô hình cộng tính (additive model):

```
y(t) = g(t) + s(t) + h(t) + r(t) + ε(t)
```

**g(t) — Trend:** Piecewise linear growth với changepoints tự động:
```
g(t) = (k + aᵀδ) × t + (m + aᵀγ)
```

**s(t) — Seasonality:** Fourier series cho tính tuần hoàn:
```
s(t) = Σ[aₙ cos(2πnt/P) + bₙ sin(2πnt/P)]
```
Với P = 365.25 (yearly), P = 7 (weekly), P = 1 (daily).

**r(t) — Regressors:** Các biến bên ngoài ảnh hưởng đến y:
```
r(t) = β₁ × humidity(t) + β₂ × cloud_cover(t)
```

### 2.7.2 Cấu hình trong đồ án

```python
model = Prophet(
    yearly_seasonality=True,   # Chu kỳ theo năm
    weekly_seasonality=True,   # Chu kỳ theo tuần
    daily_seasonality=True     # Chu kỳ theo ngày (hourly mode)
)
model.add_regressor('humidity')     # Độ ẩm ảnh hưởng nhiệt độ
model.add_regressor('cloud_cover')  # Mây che phủ ảnh hưởng nhiệt độ
```

---

## 2.8 Ensemble Learning

### 2.8.1 Lý thuyết Ensemble

Ensemble Learning kết hợp nhiều mô hình để tạo ra dự đoán tốt hơn bất kỳ mô hình đơn lẻ nào. Nguyên lý: **"Wisdom of Crowds"** — trung bình nhiều ý kiến thường chính xác hơn một ý kiến.

**Các chiến lược Ensemble phổ biến:**

| Chiến lược | Mô tả | Dùng trong đồ án? |
|---|---|---|
| Bagging | Train nhiều model trên random subsets | ❌ |
| Boosting | Train model tuần tự, sửa sai model trước | ❌ |
| Stacking | Dùng meta-learner kết hợp predictions | ❌ |
| **Weighted Averaging** | **Trung bình có trọng số** | **✅** |

### 2.8.2 Weighted Averaging trong đồ án

```
final = 0.6 × Prophet + 0.4 × LSTM
```

**Lý do chọn trọng số 60/40:**
- Prophet mạnh seasonal patterns (xu hướng dài hạn) — chiếm phần lớn biến động thời tiết
- LSTM bổ sung short-term patterns — bắt các anomaly Prophet bỏ sót
- Graceful fallback: nếu 1 model fail → dùng 100% model còn lại

### 2.8.3 Bias-Variance Trade-off

| Model | Bias | Variance | Kết hợp |
|---|---|---|---|
| Prophet | Thấp (linh hoạt) | Thấp (ít parameters) | Ổn định |
| LSTM | Thấp (deep learning) | Cao (nhiều parameters) | Có thể overfit |
| **Ensemble** | **Thấp** | **Giảm** | **Tốt nhất** |

Ensemble giảm variance bằng cách trung bình hóa — errors của 2 model triệt tiêu lẫn nhau.

---

# CHUONG 3: KIEN TRUC HE THONG

## 3.1 Tong quan kien truc

He thong duoc thiet ke theo mo hinh **Microservices Architecture** voi 5 backend services va 1 frontend:

```
+----------------------------------------------------------+
|              Dashboard UI (Port 8080 / 8501)              |
|  +-------------+  +--------------+  +----------------+   |
|  |  index.html  |  |  script.js   |  |  app.py        |   |
|  |  (Vanilla)   |  |  (Chart.js)  |  |  (Streamlit)   |   |
|  +------+-------+  +------+-------+  +-------+--------+   |
+---------+-----------------+------------------+------------+
          | HTTP            | HTTP             | HTTP
          v                 v                  v
+---------------------------------------------------------+
|              Forecast API - Orchestrator (Port 8000)      |
|                                                          |
|  predict_weather(city, mode)                             |
|    +-- get_forecast_data(city)     -> Open-Meteo API     |
|    +-- prophet_predict(df, mode)   -> Prophet model      |
|    +-- lstm_predict(temps, mode)   -> LSTM API (8004)    |
|    +-- ensemble_predictions()      -> 0.6P + 0.4L       |
+----------+-------------------------------+---------------+
           |                               |
    +------v----------+          +---------v----------+
    | Prophet API      |          | LSTM API            |
    | (Port 8003)      |          | (Port 8004)         |
    | predict_hourly   |          | predict_hourly      |
    | predict_daily    |          | predict_daily       |
    +------------------+          +---------------------+

+---------------------------------------------------------+
|              Data API (Port 8001)                        |
|  /current?city=hanoi     -> Thoi tiet hien tai           |
|  /historical?city=hanoi  -> Du lieu lich su              |
|  /forecast?city=hanoi    -> Du bao Open-Meteo            |
|  /cities                 -> Danh sach thanh pho          |
+---------------------------------------------------------+
```

## 3.2 Luong du lieu (Data Flow)

### 3.2.1 Luong du bao theo gio (Hourly)

```
1. User chon thanh pho "Ha Noi" tren Dashboard
2. Dashboard gui POST /predict {city: "hanoi", mode: "hourly", hours: 72}
3. Forecast API:
   a. Goi Open-Meteo API lay du lieu du bao 72h
   b. Goi Prophet model de du bao -> prophet_pred (72 gia tri)
   c. Goi LSTM API /predict_hourly de du bao -> lstm_pred (72 gia tri)
   d. Ket hop: final = 0.6 * prophet + 0.4 * lstm
4. Tra JSON response voi 72 diem du lieu
5. Dashboard render bieu do Chart.js
```

### 3.2.2 Luong thoi tiet hien tai

```
1. Dashboard gui GET /current?city=hanoi toi Data API
2. Data API goi Open-Meteo Forecast API (current data)
3. Tra ve: nhiet do, do am, may, gio, weather_code
4. Dashboard cap nhat Hero section + Conditions panel
```

## 3.3 Cau truc thu muc du an

```
ts/
+-- Dockerfile                    <- Docker image cho Python services
+-- docker-compose.yml            <- Orchestration 6 services
+-- .dockerignore                 <- Bo qua venv, cache khi build
+-- .env                          <- API keys (khong commit len Git)
+-- requirements.txt              <- Dependencies Python
+-- train_lstm_colab.py           <- Script training LSTM tren Google Colab
|
+-- models/                       <- Thu muc luu tru mo hinh da train
|   +-- prophet_hourly_ha_noi.json
|   +-- prophet_daily_ha_noi.json
|   +-- lstm_hourly_ha_noi.h5       (sau khi train tren Colab)
|   +-- lstm_hourly_scaler.pkl      (sau khi train tren Colab)
|
+-- src/                          <- Core ML Pipeline
|   +-- data_pipeline/
|   |   +-- fetch_data.py         <- Thu thap du lieu (6 thanh pho)
|   |   +-- preprocess.py         <- Tien xu ly
|   |   +-- feature_engineering.py<- Sliding window, temporal features
|   |
|   +-- models_logic/
|   |   +-- prophet_model.py      <- Train & save Prophet
|   |   +-- lstm_model.py         <- LSTM Encoder-Decoder class
|   |   +-- hybrid_ensemble.py    <- Weighted ensemble (0.6/0.4)
|   |
|   +-- training/
|       +-- train_hourly.py       <- Train Prophet hourly
|       +-- train_daily.py        <- Train Prophet daily
|       +-- train_lstm.py         <- Train LSTM (local)
|
+-- services/                     <- Microservices
    +-- data_api/main.py          <- Port 8001
    +-- forecast_api/main.py      <- Port 8000 (Orchestrator)
    +-- prophet_api/main.py       <- Port 8003
    +-- lstm_api/main.py          <- Port 8004
    +-- dashboard_ui/
        +-- index.html            <- Giao dien Vanilla JS
        +-- script.js             <- Logic ket noi APIs
        +-- style.css             <- Styles
        +-- app.py                <- Giao dien Streamlit
```

## 3.4 Giao tiep giua cac services

Tat ca services giao tiep qua **REST API (HTTP/JSON)**:

| Tu | Den | Phuong thuc | Mo ta |
|---|---|---|---|
| Dashboard | Data API (8001) | GET /current | Lay thoi tiet hien tai |
| Dashboard | Forecast API (8000) | POST /predict | Lay du bao ensemble |
| Forecast API | Open-Meteo | GET | Lay data tho |
| Forecast API | LSTM API (8004) | POST /predict_hourly | LSTM predictions |
| Forecast API | Prophet model | In-process | Prophet predictions (load truc tiep) |

## 3.5 Docker Network

Trong Docker Compose, cac services giao tiep qua mang noi bo `weather-net`:

```yaml
# Forecast API khong goi localhost:8001 ma goi ten container:
environment:
  - DATA_API_URL=http://data-api:8001
  - LSTM_API_URL=http://lstm-api:8004
```

---

# CHUONG 4: DATA PIPELINE

## 4.1 Nguon du lieu

### 4.1.1 Open-Meteo API (Nguon chinh)

Open-Meteo la API thoi tiet mien phi, khong yeu cau API key. Do an su dung 2 endpoint:

| Endpoint | URL | Muc dich |
|---|---|---|
| Archive API | archive-api.open-meteo.com/v1/archive | Du lieu lich su (toi da 3 nam) |
| Forecast API | api.open-meteo.com/v1/forecast | Du lieu du bao + thoi tiet hien tai |

**Bien so thu thap:**

| Bien | Don vi | Vai tro |
|---|---|---|
| temperature_2m | C | Bien muc tieu (y) |
| relative_humidity_2m | % | External regressor |
| cloud_cover | % | External regressor |
| weather_code | WMO code | Hien thi UI (icon, mo ta) |
| wind_speed_10m | km/h | Hien thi UI |

### 4.1.2 Danh sach thanh pho ho tro

```python
CITIES = {
    "hanoi":    {"lat": 21.0285, "lon": 105.8542, "name": "Ha Noi"},
    "hcm":      {"lat": 10.8231, "lon": 106.6297, "name": "TP. Ho Chi Minh"},
    "danang":   {"lat": 16.0544, "lon": 108.2022, "name": "Da Nang"},
    "haiphong": {"lat": 20.8449, "lon": 106.6881, "name": "Hai Phong"},
    "nhatrang": {"lat": 12.2388, "lon": 109.1967, "name": "Nha Trang"},
    "dalat":    {"lat": 11.9404, "lon": 108.4583, "name": "Da Lat"},
}
```

6 thanh pho dai dien cho cac vung khi hau khac nhau tai Viet Nam:
- **Ha Noi, Hai Phong**: Mien Bac — khi hau 4 mua ro ret
- **Da Nang**: Mien Trung — mua bao nhieu
- **Nha Trang**: Nam Trung Bo — nang nong
- **TP.HCM**: Mien Nam — 2 mua (mua/kho)
- **Da Lat**: Tay Nguyen — mat me quanh nam

## 4.2 Thu thap du lieu (fetch_data.py)

### 4.2.1 Ham fetch_historical()

Thu thap du lieu lich su hourly cho training mo hinh:

```python
def fetch_historical(days=1000, city="hanoi"):
    coords = get_city_coords(city)
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover",
        "timezone": "Asia/Ho_Chi_Minh"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return df[['ds', 'y', 'humidity', 'cloud_cover']]
```

**Ket qua**: DataFrame voi ~17,520 records (730 ngay x 24 gio).

## 4.3 Tien xu ly (preprocess.py)

### 4.3.1 Luong tien xu ly

```
Raw data tu API
    |
Rename columns: temperature_2m -> y, time -> ds
    |
Drop missing values (NaN)
    |
Sort theo thoi gian (ds ascending)
    |
Bo sung du lieu hom nay tu OpenWeatherMap (neu co API key)
    |
Clean DataFrame san sang cho training
```

**Luu y bao mat**: API key duoc luu trong file `.env` va load qua `python-dotenv`, khong hardcode trong source code.

## 4.4 Feature Engineering (feature_engineering.py)

### 4.4.1 Temporal Features

```python
def add_features(df, is_hourly=True):
    df['day_of_week'] = df['ds'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['month'] = df['ds'].dt.month              # 1-12
    if is_hourly:
        df['hour'] = df['ds'].dt.hour            # 0-23
        lag_list = [1, 2, 3, 6, 12, 24]
    else:
        lag_list = [1, 2, 3, 7]
    for lag in lag_list:
        df[f'temp_lag_{lag}'] = df['y'].shift(lag)
    return df
```

### 4.4.2 Sliding Window (cho LSTM)

Bien doi chuoi thoi gian thanh cac cap (input, output) cho LSTM:

```python
def sliding_window(data, window_size=24, forecast_horizon=72):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + forecast_horizon])
    return np.array(X), np.array(y)
```

### 4.4.3 Normalization

LSTM yeu cau du lieu duoc chuan hoa ve khoang [0, 1]:

```python
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data.reshape(-1, 1))
    return normalized.flatten(), scaler
```

Scaler duoc luu lai (`lstm_hourly_scaler.pkl`) de denormalize khi predict.

---

# CHUONG 5: MO HINH MACHINE LEARNING

## 5.1 Prophet Model

### 5.1.1 Training Pipeline

```python
# Step 1: Lay 2 nam du lieu lich su
df = fetch_historical(days=730)
# Step 2: Bo sung du lieu hom nay
df = add_today_from_owm(df)
# Step 3: Them features
df = add_features(df, is_hourly=True)
# Step 4: Train Prophet
prophet = train_prophet(df, is_hourly=True)
# Step 5: Luu model
save_prophet(prophet, "models/prophet_hourly_ha_noi.json")
```

### 5.1.2 Model Serialization

```python
from prophet.serialize import model_to_json, model_from_json

# Save
with open(path, "w") as f:
    json.dump(model_to_json(model), f)

# Load
with open(path, "r") as f:
    model = model_from_json(json.load(f))
```

## 5.2 LSTM Model

### 5.2.1 Kien truc chi tiet

```python
class LSTMWeatherModel:
    def build_model(self):
        inputs = keras.Input(shape=(self.lookback_window, 1))
        # Encoder
        encoder = layers.LSTM(128, activation='relu')(inputs)
        encoder = layers.Dropout(0.2)(encoder)
        encoder = layers.Dense(64, activation='relu')(encoder)
        # Repeat cho moi output time step
        decoder_input = layers.RepeatVector(self.forecast_horizon)(encoder)
        # Decoder
        decoder = layers.LSTM(64, activation='relu', return_sequences=True)(decoder_input)
        decoder = layers.Dropout(0.2)(decoder)
        outputs = layers.TimeDistributed(layers.Dense(1))(decoder)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
```

**Tong so parameters** (Hourly mode): ~107,905 parameters

### 5.2.2 Training tren Google Colab

File `train_lstm_colab.py` la script hoan toan doc lap (self-contained):

```python
model_hourly, scaler_hourly, history = train_lstm(
    mode="hourly",
    epochs=100,
    batch_size=32,
    city="hanoi"
)
```

**Callbacks su dung:**
- **EarlyStopping**: Dung khi val_loss khong giam sau 10 epochs
- **ReduceLROnPlateau**: Giam learning rate x0.5 khi val_loss khong giam sau 5 epochs

## 5.3 Hybrid Ensemble

### 5.3.1 Chien luoc ket hop

```python
def ensemble_predict(prophet_pred, lstm_pred,
                     prophet_weight=0.6, lstm_weight=0.4):
    if lstm_pred is None:
        return prophet_pred       # Fallback
    if prophet_pred is None:
        return lstm_pred          # Fallback
    min_len = min(len(prophet_pred), len(lstm_pred))
    return prophet_weight * prophet_pred[:min_len] + lstm_weight * lstm_pred[:min_len]
```

### 5.3.2 Tai sao Ensemble tot hon?

| Khia canh | Prophet alone | LSTM alone | Ensemble (0.6P + 0.4L) |
|---|---|---|---|
| Seasonal patterns | Xuat sac | Trung binh | Tot |
| Short-term anomalies | Yeu | Tot | Kha tot |
| Training time | Vai giay | 5-60 phut | Tong cong vai phut |
| Robustness | Cao | Can nhieu data | Cao (graceful fallback) |

---

# CHUONG 6: API SERVICES (MICROSERVICES)

## 6.1 Data API (Port 8001)

### 6.1.1 Endpoints

| Method | Path | Mo ta |
|---|---|---|
| GET | /cities | Danh sach thanh pho |
| GET | /current?city=hanoi | Thoi tiet hien tai |
| GET | /historical?city=hanoi&days=1000 | Du lieu lich su |
| GET | /forecast?city=hanoi&days=3 | Du bao Open-Meteo tho |

### 6.1.2 Response vi du - /current

```json
{
    "city": "Ha Noi",
    "temperature": 28.3,
    "humidity": 65,
    "cloud_cover": 40,
    "wind_speed": 12.5,
    "weather_code": 2,
    "time": "2026-04-21 18:00"
}
```

## 6.2 Forecast API (Port 8000) - Orchestrator

Service trung tam nhan request tu Dashboard, goi cac model services, ket hop ket qua va tra ve.

```python
@app.post("/predict")
def predict(request: ForecastRequest):
    result = predict_weather(
        hours=request.hours,
        days=request.days,
        mode=request.mode,
        city=request.city
    )
    return {
        "status": "success",
        "city": request.city,
        "mode": request.mode,
        "data": result.to_dict(orient="records")
    }
```

### 6.2.1 Daily Aggregation

Khi mode="daily", du lieu hourly duoc aggregate thanh daily (trung binh theo ngay):

```python
if mode == "daily":
    df_out['date'] = pd.to_datetime(df_out['ds']).dt.date
    df_out = df_out.groupby('date').agg({
        'prophet_pred': 'mean',
        'lstm_pred': 'mean',
        'final_pred': 'mean'
    }).reset_index()
```

## 6.3 Prophet API (Port 8003) va LSTM API (Port 8004)

| Service | Port | Input | Output |
|---|---|---|---|
| Prophet API | 8003 | DataFrame (ds, humidity, cloud_cover) | predictions list |
| LSTM API | 8004 | sequences (normalized values) | predictions list |

LSTM API Inference Pipeline:
```
Input: [24 gia tri normalized]
    -> reshape (1, 24, 1)
    -> lstm_hourly.predict()
    -> reshape (1, 72)
Output: [72 gia tri normalized] -> denormalize boi Forecast API
```

## 6.4 CORS Configuration

Tat ca services deu bat CORS de Dashboard co the goi API:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

# CHUONG 7: GIAO DIEN DASHBOARD

## 7.1 Tong quan giao dien

| Giao dien | Cong nghe | Port | Muc dich |
|---|---|---|---|
| Atmospheric Dashboard | HTML + JavaScript + Chart.js | 8080 | Giao dien chinh, thiet ke hien dai |
| Streamlit Dashboard | Python Streamlit | 8501 | Giao dien phu, nhanh gon |

## 7.2 Atmospheric Dashboard (Vanilla JS)

### 7.2.1 Thiet ke UI/UX

Giao dien duoc thiet ke theo phong cach **Glassmorphism** voi cac dac diem:
- **Font chu**: Manrope (Google Fonts)
- **Icons**: Material Symbols Outlined (Google)
- **Mau sac**: He mau Material Design 3 voi primary color #005da7
- **Layout**: Sidebar navigation + Main content area

### 7.2.2 Cau truc layout

```
+------------+------------------------------------------+
|            |  Header: City name + API status badge     |
|  Sidebar   +------------------------------------------+
|            |                                           |
|  Logo      |  Hero Section                             |
|  City      |  +------------------------------------+   |
|  Selector  |  |  Ha Noi, VN                         |   |
|            |  |  28 do     Troi quang                |   |
|            |  |  Do am 65%  Gio 12km/h  May 40%     |   |
|  Dashboard |  +------------------------------------+   |
|  Hourly    |                                           |
|  Daily     |  +--------------------+---------------+   |
|            |  |  Chart.js          |  Conditions    |   |
|  Model     |  |  (Hourly/Daily)    |  Panel         |   |
|  Status    |  |  [Line/Bar chart]  |  Humidity      |   |
|            |  |                    |  Wind          |   |
|            |  |                    |  Cloud         |   |
+------------+--+--------------------+---------------+---+
```

### 7.2.3 Ket noi API (script.js)

**Lay thoi tiet hien tai:**

```javascript
async function loadCurrentWeather(city) {
    const res = await fetch(`${DATA_API}/current?city=${city}`);
    const data = await res.json();
    document.getElementById("heroTemp").textContent = `${Math.round(data.temperature)} do`;
    document.getElementById("condHumidity").textContent = `${data.humidity}%`;
    document.getElementById("condWind").textContent = `${data.wind_speed} km/h`;
}
```

**Lay du bao va ve bieu do:**

```javascript
async function loadForecast(city, mode) {
    const res = await fetch(`${FORECAST_API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ city, mode, hours: 72 })
    });
    const json = await res.json();
    renderChart(json.data, mode);
}
```

### 7.2.4 Bieu do Chart.js

| Mode | Loai bieu do | Dac diem |
|---|---|---|
| Hourly | Line chart | Gradient fill, smooth tension 0.4 |
| Daily | Bar chart | Border radius 8px, semi-transparent |

### 7.2.5 WMO Weather Code Mapping

Chuyen doi ma thoi tiet WMO sang mo ta tieng Viet:

```javascript
const WEATHER_CODES = {
    0:  { icon: "light_mode",   desc: "Troi quang" },
    1:  { icon: "light_mode",   desc: "It may" },
    2:  { icon: "cloud",        desc: "Co may" },
    3:  { icon: "cloud",        desc: "Nhieu may" },
    45: { icon: "foggy",        desc: "Suong mu" },
    61: { icon: "rainy",        desc: "Mua nhe" },
    63: { icon: "rainy",        desc: "Mua vua" },
    65: { icon: "rainy",        desc: "Mua to" },
    95: { icon: "thunderstorm", desc: "Giong bao" },
};
```

## 7.3 Streamlit Dashboard

```python
FORECAST_API_URL = os.environ.get("FORECAST_API_URL", "http://localhost:8000")

city = st.selectbox("Chon thanh pho", options=[...])
tab1, tab2 = st.tabs(["Theo gio", "Trung binh 3 ngay"])

with tab1:
    response = requests.post(f"{FORECAST_API_URL}/predict",
                             json={"city": city, "mode": "hourly"})
    st.line_chart(df['final_pred'])
```

---

# CHUONG 8: DOCKER VA CONTAINERIZATION

## 8.1 Dockerfile

### 8.1.1 Chien luoc: Mot Dockerfile dung chung

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
CMD ["python", "--version"]
```

**Ly do dung chung:**
- Tat ca services dung chung requirements.txt
- Forecast API can import tu src/ (shared code)
- Docker layer cache cho pip install chi chay lai khi requirements.txt thay doi

### 8.1.2 Bien moi truong quan trong

| Bien | Gia tri | Muc dich |
|---|---|---|
| PYTHONDONTWRITEBYTECODE | 1 | Khong tao file .pyc |
| PYTHONUNBUFFERED | 1 | Print log ra ngay lap tuc |
| PYTHONPATH | /app | Python tim modules tu /app |

## 8.2 Docker Compose

### 8.2.1 Cau truc docker-compose.yml

```yaml
version: '3.8'

services:
  data-api:
    build: .
    command: uvicorn services.data_api.main:app --host 0.0.0.0 --port 8001
    ports: ["8001:8001"]
    env_file: [.env]
    networks: [weather-net]

  prophet-api:
    build: .
    command: uvicorn services.prophet_api.main:app --host 0.0.0.0 --port 8003
    ports: ["8003:8003"]
    networks: [weather-net]

  lstm-api:
    build: .
    command: uvicorn services.lstm_api.main:app --host 0.0.0.0 --port 8004
    ports: ["8004:8004"]
    networks: [weather-net]

  forecast-api:
    build: .
    command: uvicorn services.forecast_api.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    environment:
      - DATA_API_URL=http://data-api:8001
      - LSTM_API_URL=http://lstm-api:8004
    depends_on: [data-api, lstm-api, prophet-api]
    networks: [weather-net]

  dashboard-vanilla:
    image: nginx:alpine
    ports: ["8080:80"]
    volumes:
      - ./services/dashboard_ui:/usr/share/nginx/html:ro
    networks: [weather-net]

  dashboard-streamlit:
    build: .
    command: streamlit run services/dashboard_ui/app.py
             --server.port=8501 --server.address=0.0.0.0
    ports: ["8501:8501"]
    environment:
      - FORECAST_API_URL=http://forecast-api:8000
    depends_on: [forecast-api]
    networks: [weather-net]

networks:
  weather-net:
    driver: bridge
```

### 8.2.2 Giai thich

**depends_on:** Dam bao thu tu khoi dong:
```
data-api, prophet-api, lstm-api  ->  forecast-api  ->  dashboard-streamlit
```

**networks:** Tat ca services nam trong mang weather-net, giao tiep bang ten container.

### 8.2.3 Lenh van hanh

```bash
# Khoi dong toan bo he thong
docker-compose up -d --build

# Xem logs
docker-compose logs -f

# Xem logs 1 service
docker-compose logs -f forecast-api

# Dung he thong
docker-compose down

# Rebuild sau khi sua code
docker-compose up -d --build forecast-api
```

## 8.3 So do Docker Network

```
+--- Docker Host (may tinh cua ban) --------------------------+
|                                                              |
|  +-- weather-net (bridge network) ------------------------+  |
|  |                                                        |  |
|  |  +----------+  +------------+  +----------+           |  |
|  |  | data-api |  | prophet-api|  | lstm-api |           |  |
|  |  |  :8001   |  |   :8003    |  |  :8004   |           |  |
|  |  +----+-----+  +-----+------+  +----+-----+           |  |
|  |       |               |              |                 |  |
|  |       +-------+-------+--------------+                 |  |
|  |               |                                        |  |
|  |        +------v------+                                 |  |
|  |        | forecast-api|                                 |  |
|  |        |    :8000    |                                 |  |
|  |        +------+------+                                 |  |
|  |               |                                        |  |
|  |     +---------+---------+                              |  |
|  |     |                   |                              |  |
|  |  +--v--------+  +------v-------+                       |  |
|  |  | nginx     |  | streamlit    |                       |  |
|  |  |  :8080    |  |   :8501      |                       |  |
|  |  +-----------+  +--------------+                       |  |
|  +--------------------------------------------------------+  |
|                                                              |
|  Ports exposed: 8000, 8001, 8003, 8004, 8080, 8501          |
+--------------------------------------------------------------+
```

---

# CHUONG 9: KET QUA VA DANH GIA

## 9.1 Ket qua dat duoc

### 9.1.1 Ve kien truc he thong

| Tieu chi | Ket qua |
|---|---|
| So microservices | 5 (data, forecast, prophet, lstm, dashboard) |
| So thanh pho ho tro | 6 |
| Containerization | Docker Compose voi 6 containers |
| API Documentation | Tu dong qua FastAPI Swagger UI |

### 9.1.2 Ve mo hinh ML

| Tieu chi | Prophet | LSTM | Ensemble |
|---|---|---|---|
| Training data | 2 nam hourly | 2 nam hourly | - |
| Training time | ~5 giay | ~5-10 phut (GPU) | - |
| Model size | ~7.5 MB (JSON) | ~1.3 MB (.h5) | - |
| Inference time | ~200ms | ~100ms | ~300ms |
| Seasonal capture | Xuat sac | Trung binh | Tot |

### 9.1.3 Ve giao dien

| Tieu chi | Ket qua |
|---|---|
| Design | Glassmorphism, Material Design 3 |
| Responsive | Desktop + Mobile |
| Real-time data | Cap nhat tu Open-Meteo API |
| Chart | Chart.js (line + bar) |

## 9.2 Han che

1. LSTM model chua train multi-city: Hien tai chi co model cho Ha Noi.
2. Khong co CI/CD pipeline: Chua tich hop GitHub Actions.
3. Khong co model monitoring: Chua theo doi model performance.
4. Trong so ensemble co dinh: Chua tu dong dieu chinh 0.6/0.4.

## 9.3 Danh gia theo tieu chi MLOps

| Cap do MLOps | Mo ta | Do an dat? |
|---|---|---|
| Level 0: Manual | Chay notebook thu cong | Da vuot qua |
| Level 1: ML Pipeline | Automated training pipeline | Dat |
| Level 2: CI/CD Pipeline | Automated build/deploy | Co Docker, chua co CI/CD |
| Level 3: Automated Retraining | Tu train lai khi data drift | Chua dat |

Do an dat **MLOps Level 1-2**.

---

# CHUONG 10: KET LUAN VA HUONG PHAT TRIEN

## 10.1 Ket luan

Do an da xay dung thanh cong mot he thong MLOps du bao thoi tiet cho Viet Nam voi cac thanh phan:

1. **Data Pipeline**: Tu dong thu thap du lieu tu Open-Meteo API cho 6 thanh pho.
2. **Hybrid ML Model**: Ket hop Prophet va LSTM theo chien luoc Ensemble (0.6/0.4).
3. **Microservices Architecture**: 5 services doc lap giao tiep qua REST API.
4. **Containerization**: Docker Compose quan ly 6 containers.
5. **Dashboard tuong tac**: Giao dien web hien dai hien thi du bao theo gio va theo ngay.

## 10.2 Huong phat trien

### 10.2.1 Ngan han
- Train LSTM cho tat ca thanh pho
- Tich hop CI/CD voi GitHub Actions
- Them unit tests

### 10.2.2 Trung han
- Model Monitoring voi Prometheus + Grafana
- Feature Store
- A/B Testing giua cac phien ban model
- Auto-tuning ensemble weights

### 10.2.3 Dai han
- Automated Retraining khi phat hien data drift
- Multi-variable forecasting (nhiet do, mua, gio,...)
- Trien khai cloud voi Kubernetes
- Mobile app

---

# TAI LIEU THAM KHAO

1. Google. (2015). "Hidden Technical Debt in Machine Learning Systems". NeurIPS.
2. Meta/Facebook. (2017). "Prophet: Forecasting at Scale". https://facebook.github.io/prophet/
3. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory". Neural Computation.
4. Open-Meteo. (2024). "Free Weather API". https://open-meteo.com/
5. FastAPI Documentation. https://fastapi.tiangolo.com/
6. Docker Documentation. https://docs.docker.com/
7. TensorFlow/Keras Documentation. https://www.tensorflow.org/
8. Chart.js Documentation. https://www.chartjs.org/

---

# PHU LUC

## Phu luc A: Huong dan cai dat

```bash
# 1. Clone du an
git clone <repo-url>
cd ts/

# 2. Tao file .env
echo "OPENWEATHER_API_KEY=your_key_here" > .env

# 3. Chay voi Docker (khuyen nghi)
docker-compose up -d --build

# 4. Truy cap
# Dashboard:     http://localhost:8080
# Streamlit:     http://localhost:8501
# Forecast API:  http://localhost:8000/docs
# Data API:      http://localhost:8001/docs
```

## Phu luc B: Huong dan train LSTM tren Google Colab

```
1. Mo Google Colab (https://colab.research.google.com/)
2. Upload file train_lstm_colab.py
3. Chay tat ca cells tu tren xuong
4. Download file .h5 va .pkl ve may
5. Copy vao thu muc ts/models/
```

## Phu luc C: API Endpoints Reference

| Service | Port | Method | Path | Mo ta |
|---|---|---|---|---|
| Data API | 8001 | GET | /cities | Danh sach thanh pho |
| Data API | 8001 | GET | /current?city= | Thoi tiet hien tai |
| Data API | 8001 | GET | /historical?city=&days= | Du lieu lich su |
| Data API | 8001 | GET | /forecast?city=&days= | Du bao Open-Meteo |
| Forecast API | 8000 | POST | /predict | Du bao ensemble |
| Prophet API | 8003 | POST | /predict_hourly | Prophet hourly |
| LSTM API | 8004 | POST | /predict_hourly | LSTM hourly |
| LSTM API | 8004 | POST | /predict_daily | LSTM daily |
