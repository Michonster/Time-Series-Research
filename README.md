# NVIDIA Stock Price Prediction with Time Series Models

This project focuses on predicting NVIDIA's stock prices using various time series forecasting models, including RNN, LSTM, GRU, ARIMA, and SARIMA. We use historical stock data from 2010 to 2024 to train and evaluate these models, applying various fine-tuning techniques and metrics to assess performance. The project is part of an exploration into various models for stock price forecasting, with the goal of achieving highly accurate predictions using both deep learning and statistical techniques.

## Project Overview

### Goal
The primary goal of this project is to create a robust and accurate time series forecasting model for predicting NVIDIA's stock prices using a combination of machine learning and statistical approaches. We have implemented Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs). Additionally, we have explored ARIMA and SARIMA models for their effectiveness in capturing seasonality and trends.

### Accomplished So Far:
1. **Data Acquisition**: 
   - Used `yfinance` to download NVIDIA stock data from 2010 to 2024.
   - Scaled the data using Min-Max Scaler for better model performance.
   - Split the dataset into training and test sets (2010-2021 for training, 2021-2024 for testing).

2. **RNN Model**:
   - Built a Recurrent Neural Network (RNN) for time series prediction using stock price data.
   - **Final RNN Parameters**:
     - `n_steps = 30`
     - `units = 150`
     - `dropout_rate = 0.3`
     - `learning_rate = 0.00001`
     - `epochs = 200`
   - **Performance**: 
     - MSE: 9.88
     - MAE: 2.00
     - RMSE: 3.14
     - MAPE: 4.51%
   - **Conclusion**: The RNN performed well with an overall MAPE of 4.51%, and was fine-tuned to near-optimal performance.

3. **LSTM Model**:
   - Built an LSTM model to capture long-term dependencies in stock price movements.
   - **Final LSTM Parameters**:
     - `n_steps = 30`
     - `units = 300`
     - `dropout_rate = 0.3`
     - `learning_rate = 0.0005`
     - `epochs = 200`
   - **Performance**: 
     - MSE: 14.36
     - MAE: 2.48
     - RMSE: 3.79
     - MAPE: 5.88%
   - **Conclusion**: The LSTM showed promise but required further fine-tuning to match the RNN's performance.

4. **GRU Model**:
   - Implemented a Gated Recurrent Unit (GRU) model for time series prediction.
   - **Performance**: 
     - MSE: 13.74
     - MAE: 2.22
     - RMSE: 3.71
     - MAPE: 4.57%
   - **Conclusion**: The GRU model showed potential but fell short of outperforming the RNN.

5. **ARIMA Model**:
   - Built an ARIMA model to capture trends in stock prices.
   - **Final ARIMA Parameters**:
     - `(p, d, q) = (1, 1, 1)`
     - Applied logarithmic transformation to stabilize variance.
   - **Performance**:
     - MSE: 0.5468
     - MAE: 0.2320
     - RMSE: 0.7395
     - MAPE: 1.64%
   - **Conclusion**: The ARIMA model performed exceptionally well after applying logarithmic transformation, achieving the lowest error metrics.

6. **SARIMA Model**:
   - Implemented a Seasonal ARIMA (SARIMA) model to incorporate seasonality.
   - **Final SARIMA Parameters**:
     - `(p, d, q) x (P, D, Q, s) = (1, 1, 1) x (1, 1, 1, 12)`
   - **Performance**:
     - MSE: 0.5526
     - MAE: 0.2331
     - RMSE: 0.7434
     - MAPE: 1.68%
   - **Conclusion**: The SARIMA model captured seasonal patterns well but did not significantly outperform ARIMA.

### What's Left To Do (Checklist)

1. **Evaluation Across Datasets**:
   - [ ] Test the RNN, LSTM, GRU, ARIMA, and SARIMA models on two additional datasets to ensure robustness and avoid sampling bias.

2. **Explore Advanced Models**:
   - [ ] Investigate using Attention-based models, which may improve prediction accuracy by focusing on important time steps.
   - [ ] Experiment with Hybrid models that combine multiple architectures for better predictions.

3. **Deployment**:
   - [ ] Deploy the best-performing model for real-time stock price prediction.

### Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/nvidia-stock-prediction.git
   cd nvidia-stock-prediction
2. **Install required packages**: Use the requirements.txt file to install all dependencies.
   ```bash
   pip install -r requirements.txt
3. **Download the data**: Ensure you have the NVIDIA stock data downloaded using the yfinance library.
4. **Run the Jupyter Notebook**: Launch the project and start training the models by running the cells in the provided notebook.
   ```bash
   jupyter notebook

### Models in Use

- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Autoregressive Integrated Moving Average (ARIMA)
- Seasonal ARIMA (SARIMA)
- Attention-Based Models (Planned)
- Hybrid Models (Planned)

### Performance Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Results
| Model | MSE  | MAE  | RMSE | MAPE  |
|-------|------|------|------|-------|
| RNN   | 9.88 | 2.00 | 3.14 | 4.51% |
| LSTM  | 14.36 | 2.48 | 3.79 | 5.88% |
| GRU	  | 13.74 |	2.22 | 3.71	| 4.57% |
| ARIMA	| 0.5468	| 0.2320	| 0.7395 |1.64% |
| SARIMA	| 0.5526	| 0.2331	| 0.7434	| 1.68% |

### Future Work
- Further fine-tuning of LSTM and GRU models.
- Implement Attention and Hybrid models.
- Explore real-time stock prediction using the best-performing model.

### Contributors
- **Research Assistant**: Shane Stoll
- **Researcher**: Uzma Mushtaque
