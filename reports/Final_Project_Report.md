# Time-Based Browsing Pattern Analyzer: Final Project Report

## Project Overview

This project analyzes user browser activity and system RAM consumption over time to discover behavioral patterns, segment user sessions, forecast future next-session categories using deep learning (PyTorch LSTM), and deliver prediction-driven recommendations.

All modeling stages maintain strict experiment tracking using **MLflow**, storing metrics, model checkpoints, parameters, and loss artifacts.

---

## 1. Executive Summary & Data Overview

The underlying dataset integrates user Chrome browsing history synchronized with continuous 5-second interval RAM usage telemetry.

### Key Data Metrics:
- **Total Browsing Events**: 24,321 events
- **Total Sessions Analyzed**: 829 sessions
- **Observed Categories**: 16 distinct categories

### Top 10 Domains by Event Frequency
| Domain | Category | Event Count |
| :--- | :--- | :--- |
| `google.com` | Search/Reference | 10,227 |
| `instagram.com` | Social Media | 4,301 |
| `colab.research.google.com` | Learning/Education | 1,508 |
| `m.youtube.com` | Social Media | 861 |
| `github.com` | Learning/Education | 545 |
| `accounts.google.com` | Search/Reference | 494 |
| `linkedin.com` | Social Media | 234 |
| `docs.google.com` | Productivity/Work | 225 |
| `drive.google.com` | Productivity/Work | 214 |
| `youtube.com` | Social Media | 210 |

![Top 10 Visited Domains](file:///home/abishek/Projects/intelligent-browser-behaviour-analytics-platform/reports/images/top_domains.png)

---

## 2. RAM Usage Correlation Analysis

Analyzing system RAM utilization across browsing categories reveals heavy memory footprints for interactive web applications and video streaming.

### RAM Statistics by Category
| Category | Mean Used RAM (MB) | Peak Used RAM (MB) | Peak Usage (%) |
| :--- | :--- | :--- | :--- |
| Search/Reference | 3,477.11 | 6,659.90 | 81.3% |
| Social Media | 3,613.28 | 6,655.10 | 81.2% |
| Productivity/Work | 3,482.92 | 6,212.90 | 75.8% |
| Shopping | 3,637.20 | 6,163.40 | 75.2% |
| Piracy/Streaming | 3,685.98 | 6,129.20 | 74.8% |
| Entertainment/Media | 3,564.58 | 6,075.30 | 74.2% |
| Learning/Education | 3,435.59 | 5,708.70 | 69.7% |
| Job Search/Career | 3,448.37 | 5,368.70 | 65.5% |

![Peak RAM Usage by Category](file:///home/abishek/Projects/intelligent-browser-behaviour-analytics-platform/reports/images/category_ram_correlation.png)

---

## 3. Session Behavior Segmentation (KMeans Clustering)

Unsupervised **KMeans Clustering** applied to standardized session behavior features (duration, page count, category switching, peak RAM, time of day) identified **2 primary session profiles**:

### Behavior Clusters Profile
- **Cluster 0 (Low Memory / Short Focus)**: Average duration ~13.5 mins, 29.6 pages, mean RAM 3,443 MB.
- **Cluster 1 (High Memory / Heavy Workload)**: Average duration ~11.0 mins, 21.5 pages, mean RAM 5,354 MB, peak RAM ~5,780 MB (70.5% RAM pressure).

![Session Clusters PCA](file:///home/abishek/Projects/intelligent-browser-behaviour-analytics-platform/reports/images/session_clusters.png)

---

## 4. Deep Learning: Next-Session Category Forecasting

A PyTorch **Multi-Task LSTM Neural Network** forecasts the user's next session category and numerical load based on a 5-session historical window.

### Model Architecture:
- **Embedding Layer**: 128-dimensional category embeddings
- **LSTM Backbone**: 2 Recurrent Layers, 256 Hidden Units, 0.5 Dropout
- **Output Heads**:
  - Softmax Classification Head for Category prediction
  - Linear Regression Head for continuous session metrics forecast
- **Training Setup**: Adam Optimizer (lr=0.001), 20+ Epochs, logged to MLflow.

---

## 5. Prediction-Aware Actionable Recommendations

The recommendation engine combines prediction forecasts with behavioral thresholds to produce contextual recommendation cards:

1. **Late Night Social Browsing Alert** (Medium 🟠): Late-night sessions starting after 22:00 dominated by social media.
2. **Heavy RAM Usage Warning** (High 🔴): Session peak memory exceeds 75% threshold; recommends closing idle tabs.
3. **Task Fragmentation Notice** (Medium 🟠): High category switching frequency indicates multi-tasking distraction.
4. **Dominant Site Engagement** (Low 🟢): `google.com` accounts for >30% of total visits.

---

## 6. Deliverable Manifest

- `data/gold/session_features.parquet`: Engineered gold session features
- `data/gold/session_segments.parquet`: Clustered session segment labels
- `data/gold/behavior_sequences.parquet`: 5-session sliding window sequences
- `models/lstm_next_session.pt`: Trained PyTorch LSTM model checkpoint
- `models/kmeans_clustering.pkl`: Persisted KMeans segmenter model
- `reports/images/`: Visualizations (`session_clusters.png`, `category_ram_correlation.png`, `top_domains.png`, `hourly_heatmap.png`)
- `mlruns/`: Centralized MLflow tracking database & artifacts
