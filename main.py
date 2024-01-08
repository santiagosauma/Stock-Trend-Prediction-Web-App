import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set page layout to wide
st.set_page_config(layout="wide")

# CSS to inject contained in a string
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        /* Updated styles here */
        background: radial-gradient(circle at top left, transparent 9%, #cccccc 10%, #cccccc 15%, transparent 16%),
                    radial-gradient(circle at bottom left, transparent 9%, #cccccc 10%, #cccccc 15%, transparent 16%),
                    radial-gradient(circle at top right, transparent 9%, #cccccc 10%, #cccccc 15%, transparent 16%),
                    radial-gradient(circle at bottom right, transparent 9%, #cccccc 10%, #cccccc 15%, transparent 16%),
                    radial-gradient(circle, transparent 25%, #ffffff 26%),
                    linear-gradient(0deg, transparent 44%, #cccccc 45%, #cccccc 55%, transparent 56%),
                    linear-gradient(90deg, transparent 44%, #cccccc 45%, #cccccc 55%, transparent 56%);
        background-size: 3em 3em;
        background-color: #ffffff;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define START and TODAY for fetching data
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App title
st.title("Stock Prediction App")

# User input for the stock query
selected_stock = st.text_input("Write the Company actions:")

if selected_stock:
    # Slider for years of prediction
    n_years = st.slider("Years of prediction:", 1, 4, 1)
    period = n_years * 365

    # Function to load data
    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Load and cache data
    data_load_state = st.text("Load data...")
    data = load_data(selected_stock)
    data_load_state.text("Loading data...done!")

    # First row: Raw data table and plot
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Raw data')
        st.write(data.tail())
    with col2:
        st.subheader('Time Series Data')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    # Second row: Forecast plot and forecast data table
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Forecast plot')
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1, use_container_width=True)
    with col4:
        st.subheader('Forecast data')
        st.write(forecast.tail())

    # Forecast components at full width below both columns
    st.subheader('Forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
