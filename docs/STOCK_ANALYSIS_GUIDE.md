# Stock Analysis Primer

A guide to the basic concepts of stock analysis mentioned in this project.

---

## Three Main Types of Analysis

Stock analysis is generally categorized into three main areas. This project primarily focuses on **Technical** and **Quantitative** analysis, but it's useful to understand all three.

### 1. Fundamental Analysis

**What it is:** The practice of evaluating a company's financial health and market position to determine its intrinsic value.

**Simple Analogy:** It's like being a detective for a business. You read its financial reports (like the `income statement` and `balance sheet`), assess its leadership, and understand its competitive advantages.

**Key Questions:**
- Is the company profitable?
- Does it have a lot of debt?
- Is it a leader in its industry?
- Is the stock price cheap or expensive compared to its earnings (P/E Ratio)?

This project uses fundamental data in some of its more advanced features.

### 2. Technical Analysis

**What it is:** The study of historical price charts and trading volumes to predict future price movements. It assumes that all important information is already reflected in a stock's price.

**Simple Analogy:** It's like being a weather forecaster. You look at past weather patterns (charts) to predict if it will rain tomorrow (if the stock will go up or down).

**Key Concepts Mentioned in the `ML_MODULE_GUIDE.md`:**

| Term | What it Measures | Simple Explanation |
|---|---|---|
| **RSI (Relative Strength Index)** | Momentum | Tells you if a stock is "overbought" (might be too expensive) or "oversold" (might be too cheap). It's like a speedometer for the stock price. |
| **MACD (Moving Average Convergence Divergence)** | Trend Direction & Momentum | Helps identify if a stock's trend is getting stronger or weaker. It shows the relationship between two moving averages of a stock's price. |
| **Bollinger Bands** | Volatility | Two bands that are drawn above and below a moving average. When the bands are wide, the stock is volatile. When they are narrow, it's less volatile. Prices are considered high at the upper band and low at the lower band. |
| **Moving Average** | Trend Direction | The average price of a stock over a specific period (e.g., 50 days or 200 days). It helps smooth out price action and identify the direction of the trend. |

### 3. Quantitative Analysis ("Quant")

**What it is:** The use of mathematical models, statistical techniques, and computational power to analyze financial data. This is exactly what the `ml/` module in this project does.

**Simple Analogy:** It's like building a team of robots to do technical and fundamental analysis at a massive scale, looking for patterns that a human might miss.

**How it works in this project:**
1.  **Data Input**: The system takes in vast amounts of data (prices, volume, etc.).
2.  **Feature Engineering**: It calculates hundreds of indicators (like RSI, MACD, and many more complex ones).
3.  **Modeling**: Machine learning models (the "Ensemble" in the ML guide) learn the relationships between these features and future price movements.
4.  **Prediction**: The model outputs a prediction (e.g., "70% chance the stock will go up in the next 5 days").

---

## Other Important Concepts

### Sentiment Analysis

**What it is:** The process of gauging the overall mood of investors and the public towards a particular stock. This can be done by analyzing news articles, social media posts, and financial reports.

**Simple Analogy:** It's like listening to the "buzz" around a stock. Is everyone excited about it, or are they fearful?

### Risk Management

**What it is:** The process of identifying, assessing, and mitigating risks in an investment portfolio.

**Key Concepts:**
| Term | What it Measures | Simple Explanation |
|---|---|---|
| **VaR (Value at Risk)** | Potential Loss | A statistical measure that estimates the maximum potential loss a portfolio could experience over a given time period, with a certain level of confidence. For example, "There is a 95% confidence that the portfolio will not lose more than $10,000 in the next trading day." |
| **Position Sizing** | How much to invest | Deciding how much capital to allocate to a single investment. A common rule is to not risk more than 1-2% of your total portfolio on a single trade. |
| **Diversification** | Spreading out investments | The practice of investing in a variety of assets to reduce the impact of a single asset's poor performance on the overall portfolio. "Don't put all your eggs in one basket." |

---

This guide provides a basic overview. Each of these topics is a deep field of study, but this should give you enough context to understand the different components of the AI Stock Analyst project.
