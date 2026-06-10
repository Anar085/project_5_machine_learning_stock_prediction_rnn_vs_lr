# RNN vs Logistic Regression - Stock Price Movement Prediction

A comparative study examining whether Recurrent Neural Networks (LSTMs) actually outperform traditional Logistic Regression models when predicting daily stock price movements.

## Background

There's an ongoing debate in quantitative finance about whether complex deep learning models are worth the overhead compared to simpler statistical methods. This paper takes a hands-on approach: train both models on real market data, run them against the same stocks, and let the numbers speak.

## What's in the Paper

The paper covers the theoretical foundations of both models - how LSTM gates work, how logistic regression uses lagged returns and moving averages as features - before moving into the actual experiment.

**Data:** 4 years of daily price data (2021–2024) across 10 stocks from 5 sectors, deliberately mixing large-cap and small/mid-cap companies:

| Sector | Large Cap | Small/Mid Cap |
|---|---|---|
| Technology | MSFT | AMBA |
| E-commerce | AMZN | SFTX |
| Healthcare | JNJ | BLUE |
| Finance | JPM | LC |
| Oil & Gas | XOM | SM |

**Models:** LSTM (via TensorFlow/Keras with KerasTuner hyperparameter optimization) vs. Logistic Regression (scikit-learn) with engineered features - return lags, moving averages, and volatility.

## Key Findings

Neither model dominates cleanly. LSTM edges out LR on accuracy and precision, while LR wins on recall and F1. A few patterns that stood out:

- LSTM performs better on **volatile, complex stocks** (Tech, E-commerce, smaller caps)
- LR holds up better on **steadier, lower-volatility stocks** (Finance, Oil & Gas)
- LR is dramatically faster - roughly **1000x less training time** for marginal performance differences
- LSTM showed clear **overfitting** on short-horizon equity data in several stocks
- LR's feature coefficients provide interpretability that LSTM simply can't match

## Conclusion

LSTMs are only *conditionally* better. If you need precise, confident signals and have compute to spare - go LSTM. If you need speed, interpretability, and good recall - LR is hard to beat. The results suggest model choice should depend on the stock type and use case, not complexity for its own sake.

## Stack

Python 3.12 · scikit-learn · TensorFlow/Keras · KerasTuner · Pandas · NumPy · Alpha Vantage API
