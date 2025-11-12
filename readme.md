# 🧠 Machine Learning Project (Semester 5)

## **Project Title**

**Intelligent Real-Time Bidding System using Machine Learning**

---

## **Problem Statement**

Advertisers aim to maximize their **Return on Investment (ROI)** by generating more clicks and high-value conversions while operating within fixed budgets.
The key challenge is to design an **intelligent Real-Time Bidding (RTB)** system that can dynamically balance ad expenditure, conversion probability, and overall campaign performance, all while adhering to latency and budget constraints.

The system must:

* Accurately identify optimal ad placements and bid amounts.
* Continuously adapt to changing audience behaviors and campaign objectives.
* Optimize performance while respecting operational constraints.

---

## **Objective**

To design and implement a **learning-based, low-latency RTB bidding agent** capable of predicting **Click-Through Rate (CTR)** and **Conversion Rate (CVR)** using historical data and optimizing bid prices in real time to maximize advertiser ROI.

---

## **Dataset**

**Source:** *iPinYou 2013 Global RTB Bidding Algorithm Competition*

This large-scale dataset includes information on ad impressions, user interactions, and conversions from multiple advertisers and campaigns. All identifiers are **hashed/anonymized** to preserve privacy.

**Dataset Sizes:**

| Data Type       | Records    |
| --------------- | ---------- |
| Bid Data        | 53,289,330 |
| Impression Data | 12,237,087 |
| Click Data      | 9,978      |
| Conversion Data | 494        |

**Key Observations:**

* The data is sparse and highly imbalanced.
* CTR ≈ 0.08%, CVR ≈ 0.004%.
* Precision is prioritized over recall to ensure efficient ad spend optimization.

---

## **Feature Engineering**

To enhance model performance, several transformations and derived features were created:

* **Aspect Ratio:** `adslot_width / adslot_height`
* **Creative–User Affinity:** Frequency of interactions between a specific creative and user.
* **Temporal Features:** Hour, weekday, and weekend indicators.
* **Contextual Features:** Advertiser ID, device type, and site category.
* **Historical CTR/CVR:** Per advertiser or site for trend-based learning.
* **Price Ratios:** `paying_price / bidding_price` to measure auction efficiency.

High-cardinality categorical features were **target encoded**, while low-cardinality ones used **one-hot encoding**.
All numerical features were **scaled or log-transformed** to address skewness.
The **Polars** library was used for faster and memory-efficient data manipulation.

---

## **Data Preprocessing Pipeline**

1. **Data Ingestion:** Merge Bid, Impression, Click, and Conversion logs using `BidID`.
2. **Cleaning & Validation:** Remove nulls, duplicates, and invalid timestamps.
3. **Imputation:** Replace missing categorical values with `"unknown"`; numerical with median.
4. **Feature Selection:** Drop redundant or low-variance features and remove multicollinear columns (VIF > 0.9).
5. **Sampling & Split:** Use temporal split for training/validation and apply class weighting for imbalance handling.
6. **Transformation:** Use Polars `LazyFrame` joins and cache intermediate computations.
7. **Model Preparation:** Export cleaned feature matrix and targets for training and inference.

---

## **Modelling Approach**

A **stacked regression ensemble** was developed to capture both non-linear and structured patterns:

1. **Base Models**

   * **ANN:** 5-layer fully connected network — [128, 64, 32, 16, 1] — ReLU activations and 0.25 dropout; optimized with **Huber Loss**.
   * **XGBoost Regressor:** Captures monotonic dependencies and rule-based feature relations.

2. **Meta Regressor**

   * **Third-order polynomial regression** combining ANN and XGBoost outputs with campaign-level signals (ROI, pacing ratio, etc.).

3. **Federated Learning Regime**

   * Trains advertiser-specific local models and aggregates updates globally to ensure **privacy-preserving learning**.

4. **Constraint Regularization**

   * Business rules (budget caps, pacing constraints) used as regularization terms.

---

## **Model Optimization**

To meet industry-grade latency and memory constraints:

* **Pruning:** Remove low-importance neurons (~70% reduction in ANN parameters).
* **Quantization:** Convert 32-bit weights to 8-bit integers (4× compression).
* **Dynamic Feature Binning:** Replace floating-point operations with pre-learned integer bins.
* **Graph Fusion:** Combine linear and activation operations to minimize runtime overhead.

---

## **Evaluation**

### Metrics

* **Precision**, **Recall**, **Weighted F1**
* **ROC-AUC**, **RMSE**
* Custom metrics:

  * **Expected ROI Gain (ERoI)**
  * **Normalized Budget Efficiency (NBE)**
  * **Latency–Accuracy Tradeoff (LAT)**

### Results

| Advertiser ID       | ROC-AUC   | RMSE      |
| ------------------- | --------- | --------- |
| 1458                | 0.916     | 0.015     |
| 3476                | 0.880     | 0.014     |
| 3427                | 0.930     | 0.0165    |
| 3386                | 0.840     | 0.014     |
| 3358                | 0.830     | 0.021     |
| **All Advertisers** | **0.985** | **0.021** |

✅ **Average Inference Time:** 3.8 ms
✅ **High stability across campaigns**
✅ **Production-ready latency and memory efficiency**

---

## **Conclusion**

The project successfully developed a **real-time, low-latency ML bidding system** for digital advertising.
The combination of feature engineering, ensemble modelling, and constraint-aware optimization achieved:

* **High predictive accuracy (ROC-AUC = 0.985)**
* **Sub-5 ms inference latency**
* **Strong trade-off between accuracy, speed, and compactness**

This system demonstrates how **machine learning can drive intelligent, scalable, and privacy-aware ad bidding** in real-world RTB environments.

---

## **Code Usage**

To implement and test the bidder interface:

1. **Create and activate a virtual environment**

   ```bash
   python -m venv ./venv
   venv\Scripts\activate             # for Windows
   source venv/bin/activate          # for macOS/Linux
   ```

2. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Import and use the bidder**

   ```python
   from bidder.ordinal import OrdinalBidder
   from bidder.BidRequest import BidRequest

   bidding_agent = OrdinalBidder()
   ordinal_bidder_bid = bidding_agent.getBidPrice(bidRequest)
   ```

---