# NVIDIA Stock Price Prediction with Time Series Models

This project focuses on predicting NVIDIA's stock prices using various time series forecasting models, including RNN and LSTM. We use historical stock data from 2010 to 2024 to train and evaluate these models, applying various fine-tuning techniques and metrics to assess performance. The project is part of an exploration into various models for stock price forecasting, with the goal of achieving highly accurate predictions using deep learning techniques.

## Project Overview

### Goal
The primary goal of this project is to create a robust and accurate time series forecasting model for predicting NVIDIA's stock prices using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. Future steps will involve experimenting with other advanced models like Gated Recurrent Units (GRUs) and hybrid models that combine the strengths of multiple architectures.

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
   - **Current LSTM Parameters**:
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
   - **Conclusion**: While the LSTM showed promise, its performance was slightly inferior to the RNN, requiring further fine-tuning.

### What's Left To Do (Checklist)

1. **Fine-Tuning LSTM**:
   - [ ] Experiment with different `n_steps` values (e.g., 60, 90).
   - [ ] Adjust dropout rate (e.g., reduce to 0.1 or 0.2).
   - [ ] Test with more LSTM units (e.g., 350-400 units).
   - [ ] Experiment with batch size to see its effect on learning stability.

2. **GRU Model**:
   - [ ] Implement and test a Gated Recurrent Unit (GRU) model.
   - [ ] Evaluate the GRU model’s performance compared to RNN and LSTM.
   - [ ] Fine-tune hyperparameters for the GRU model.

3. **Explore Advanced Models**:
   - [ ] Investigate using Attention-based models, which may improve prediction accuracy by focusing on important time steps.
   - [ ] Experiment with Hybrid models that combine multiple architectures for better predictions.

4. **Evaluate Performance on Other Metrics**:
   - [ ] Try using other error metrics such as Symmetric Mean Absolute Percentage Error (SMAPE) to evaluate models.
   
5. **Deploy the Best Model**:
   - [ ] Once a model is finalized, work on deployment for real-time stock price prediction.

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
- Gated Recurrent Units (GRU) (Upcoming)
- Autoregressive Integrated Moving Average (ARIMA) (Planned)
- Attention-Based Models (Planned)
- Hybrid Models (Planned)

### Performance Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Results


### Future WOrk
- Further fine-tuning of LSTM and GRU models.
- Implement Attention and Hybrid models.
- Explore real-time stock prediction using the best-performing model.

### Contributors
- **Research Assistant**: Shane Stoll
- **Researcher**: Uzma Mushtaque
