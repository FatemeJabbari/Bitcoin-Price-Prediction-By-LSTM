📈 Bitcoin Price Prediction Using LSTM

🎓 Overview

This repository implements a Long Short-Term Memory (LSTM) neural network model to predict Bitcoin prices using historical price data. LSTM networks are a type of Recurrent Neural Network (RNN) especially well-suited for time-series forecasting due to their ability to learn long-term dependencies in sequential data. Models like this are widely used in financial forecasting research and applications. 
GitHub

🧠 Objective

The goal of this project is to:

Prepare Bitcoin price data for supervised learning

Train an LSTM model to learn temporal patterns in price movements

Evaluate its predictive performance

Visualize actual vs. predicted price trends

This approach helps understand the predictive ability of deep learning models in volatile markets such as cryptocurrency.

📁 Repository Structure
Bitcoin-Price-Prediction-By-LSTM/
├── BitcoinPricePredictio2.ipynb   # Main prediction notebook
├── outputcode.png                 # Example output figure
└── README.md                     # (This file)


📌 Add any dataset files or additional notebooks under clearly named folders if needed (e.g., data/).

📊 Methodology
✅ 1. Data Loading and Preprocessing

Load historical Bitcoin price data (e.g., daily closing prices)

Normalize the data for better neural network performance

Scaling between 0–1 (e.g., MinMaxScaler)

Structure data into look-back sequences to be used for training

✅ 2. LSTM Model Architecture

A network of one or more LSTM layers

Dense output layer for price prediction

Loss: Mean Squared Error (MSE)

Optimizer: Adam

LSTM models are effective for sequential prediction problems because they can retain information from prior steps in the sequence, unlike standard feed-forward networks. 
GitHub

📈 Visualizations
🟦 Actual vs Predicted Prices


A common visualization used in forecasting projects is a comparison between the real prices and the model’s predicted values:

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")
plt.title("Bitcoin Price Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Price (Normalized / Real)")
plt.legend()
plt.show()


This plot helps evaluate how well the model tracks the overall price trend. 
GitHub

🟩 Loss Curve (Training)

Plotting training loss over epochs shows model learning behavior:

plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

📌 How to Run
🪶 Prerequisites

Make sure you have:

Python 3.7+

Jupyter Notebook or JupyterLab

Installed packages such as numpy, pandas, matplotlib, tensorflow or keras

🛠 Installation

Clone the repository

git clone https://github.com/FatemeJabbari/Bitcoin-Price-Prediction-By-LSTM
cd Bitcoin-Price-Prediction-By-LSTM


Install dependencies

pip install numpy pandas matplotlib tensorflow scikit-learn


Open and run the notebook

jupyter notebook BitcoinPricePredictio2.ipynb


➡️ Follow sequential steps in the notebook to:

Load data

Preprocess and normalize

Build and train the LSTM

Generate predictions

Visualize results

📊 Performance Metrics (Optional)

Model performance can be assessed using typical regression metrics:

Metric	Description
MSE	Mean Squared Error – average squared difference between prediction and truth
RMSE	Root Mean Squared Error – square root of MSE
MAE	Mean Absolute Error – average absolute difference

Add these to quantify forecasting accuracy.

📌 Notes & Tips

Accuracy depends on the look-back window and network architecture.

Cryptocurrency markets are highly volatile — predictions should be interpreted with caution.

Features beyond price (volume or technical indicators) can improve forecasting results.

📚 References

Example implementations of Bitcoin LSTM forecasting models from GitHub (comparable workflows and visualizations). 
GitHub

Time-series LSTM best practices and model evaluation guidelines.

🚀 Next Steps (Enhancements)

You can improve this project by:

Adding technical indicators (e.g., moving averages, RSI)

Experimenting with more layers or Bidirectional LSTM

Using multivariate inputs

Adding train/test split visualizations

Saving and exporting the trained model for real-time prediction
