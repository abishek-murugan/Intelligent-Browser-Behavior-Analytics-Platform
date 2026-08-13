# PyTorch LSTM Next-Session Forecasting Report

## Overview
The deep learning forecasting pipeline uses a multi-task PyTorch Long Short-Term Memory (LSTM) network to predict the user's next browsing session profile based on a sequence of 5 historical sessions.

## Network Parameters
- Sequence Length ($L$): 5 sessions
- Category Embedding Size: 128
- LSTM Layers: 2 layers
- Hidden State Dimension: 256 units
- Dropout Rate: 0.5
- Optimization: Adam (Learning Rate = 0.001)

## MLflow Tracking
Training progress, loss metrics, and PyTorch model state dicts are saved under experiment `browser-behavior-lstm`.
