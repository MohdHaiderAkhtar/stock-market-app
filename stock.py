import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import pipeline
import plotly.graph_objects as go

# Initialize BERT sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# API keys
NEWS_API_KEY = "your_newsapi_key"

# Title of the app
st.title("Advanced Stock Market Analysis App")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):")

# Date range selection
start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2022-12-31"))

if ticker:
    # Fetch historical stock data
    st.header("Historical Stock Data")
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write(f"Showing data for {ticker}")
    st.dataframe(data)

    # Fetch news articles
    st.header("News Sentiment Analysis")
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    news_data = response.json()

    sentiment_scores = []
    if news_data["status"] == "ok" and news_data["totalResults"] > 0:
        articles = news_data["articles"]
        for article in articles[:5]:  # Analyze top 5 articles
            sentiment = sentiment_analyzer(article['description'] or "")
            sentiment_scores.append(sentiment[0]['score'] if sentiment[0]['label'] == "POSITIVE" else -sentiment[0]['score'])
            st.write(f"**{article['title']}**")
            st.write(f"Sentiment: {sentiment[0]['label']} (Score: {sentiment[0]['score']:.2f})")
            st.write("---")
    else:
        st.error("No news articles found for this ticker.")

    # Add average sentiment score as a feature
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    data['Sentiment'] = average_sentiment

    # Prepare data for machine learning
    st.header("Stock Price Prediction")
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    X = data[['Close', 'Sentiment']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    st.write("Mean Squared Error:", mse)

    # Combine actual and predicted values for visualization
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})

    # Plot actual vs predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['Actual'],
        mode='lines',
        name='Actual Prices'
    ))
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['Predicted'],
        mode='lines',
        name='Predicted Prices'
    ))
    fig.update_layout(
        title=f"{ticker} Stock Price Prediction",
        xaxis_title="Index",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

    # Visualize sentiment trends
    st.header("Sentiment Trends")
    sentiment_df = pd.DataFrame({"Sentiment Score": sentiment_scores})
    sentiment_df['Timestamp'] = pd.date_range(start=start_date, periods=len(sentiment_scores))
    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(go.Scatter(
        x=sentiment_df['Timestamp'],
        y=sentiment_df['Sentiment Score'],
        mode='lines+markers',
        name='Sentiment Score'
    ))
    fig_sentiment.update_layout(
        title=f"Sentiment Trends for {ticker}",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template="plotly_dark"
    )
    st.plotly_chart(fig_sentiment)

    # Real-time updates (optional)
    st.header("Real-Time Stock Price")
    real_time_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    st.write(f"Real-Time Price for {ticker}: {real_time_price}")