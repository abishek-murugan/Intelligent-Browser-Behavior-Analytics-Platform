# Session Behavior Clustering Report

## Methodology
The session behavior clustering module partitions browser browsing sessions into behavioral segments using unsupervised KMeans clustering on standardized temporal, volumetric, and resource utilization features.

## Feature Set:
1. `session_duration_seconds`
2. `page_count`
3. `unique_domains`
4. `unique_categories`
5. `domain_switches`
6. `category_switches`
7. `avg_usage_percent`
8. `peak_usage_percent`

## Results & Profiles
- Optimal K selected via Silhouette analysis: **K = 2**
- **Segment 0**: Moderate RAM, low switching frequency.
- **Segment 1**: High RAM usage spike, intense multitasking.

MLflow experiment runs are tracked under experiment `browser-behavior-clustering`.
