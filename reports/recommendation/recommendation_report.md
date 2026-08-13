# Browser Recommendation Engine Report

## Strategy
The recommendation engine merges model predictions with behavioral heuristics:
1. **LSTM Category Forecast**: Prioritizes categories forecasted by the deep learning model.
2. **Category Affinity**: Ranks historically high-engagement categories.
3. **Contextual Rules**: Evaluates memory alerts, late-night usage, and task switching.
4. **Severity Badges**: Tags recommendations with High (🔴), Medium (🟠), or Low (🟢) priority.

Results are persisted in `data/gold/recommendations.parquet`.
