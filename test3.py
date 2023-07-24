import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import warnings
import io
import requests
from io import StringIO



st.set_page_config(
    page_title='PM Demand Forecast',
    layout="wide"
)

# Function for SARIMA forecast and plot
def sarima_forecast(brand_df):
    # Remove rows with zero sales
    brand_df = brand_df[brand_df['Daily Sales'] != 0]

    # Handle missing values in exogenous variables
    brand_df['USD/LBP Rate'].fillna(method='ffill', inplace=True)  # Forward fill missing values
    brand_df['Price'].fillna(brand_df['Price'].mean(), inplace=True)  # Fill with mean

    # Convert 'Date' to datetime index and sort
    brand_df.set_index('Date', inplace=True)
    brand_df.sort_index(inplace=True)

    # Create a series for daily sales and exogenous variables
    sales_series = brand_df['Daily Sales']
    exog_vars = brand_df[['USD/LBP Rate', 'Price']]

    try:
        # Fit the SARIMAX model with exogenous variables
        model = SARIMAX(sales_series, exog=exog_vars, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))  # Set the seasonal_order as needed
        model_fit = model.fit()

        # Forecast for 365 days into the future from the last available date in the data
        last_date = sales_series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=365, freq='D')
        forecast = model_fit.forecast(steps=365, exog=exog_vars.tail(365))

        # Plot the forecast using Plotly
        fig = go.Figure()

        # Actual Sales trace
        fig.add_trace(go.Scatter(x=sales_series.index, y=sales_series,
                                 mode='lines+markers',
                                 name='Actual Sales',
                                 line=dict(color='#0062A3', width=2, shape='spline', smoothing=0.5)))  # Smoother line

        # Forecast trace
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast,
                                 mode='lines+markers',
                                 name='Forecast',
                                 line=dict(color='#1CFB05', width=2, dash='dot', shape='spline', smoothing=0.5)))  # Smoother line

        # Layout settings
        fig.update_layout(title=f'Sales Forecast for {selected_brand}',
                          xaxis_title='',
                          yaxis_title='',
                          showlegend=True,
                          legend=dict(x=1.0, y=1.0, bgcolor='rgba(255,255,255,0.5)'),  # Move legend to the right
                          plot_bgcolor='white',
                          xaxis_showgrid=False,  # Removed gridlines on x-axis
                          yaxis_showgrid=False,  # Removed gridlines on y-axis
                          margin=dict(l=50, r=50, t=50, b=50),
                          xaxis=dict(
                              tickformat='%Y',  # Format x-axis to display only years
                              dtick='M12'  # Set tick interval to 1 year
                          ),
                          yaxis=dict(
                              showticklabels=False,  # Hide y-axis tick labels
                              showgrid=False  # Hide y-axis grid lines
                          ))

        # Update the trace styles to match fbprophet's plot style
        fig.update_traces(marker=dict(size=4, opacity=0.7), selector=dict(mode='markers'))  # Smaller marker size and opacity

        return fig, mean_absolute_percentage_error(sales_series, model_fit.fittedvalues), forecast_dates, forecast
    except ValueError as e:
        raise ValueError(f"Skipping forecast for {selected_brand} due to the following error: {e}")


# Function to generate insights
def generate_insights(brand_df, forecast_dates, forecast):
    insights = []

    # Basic trend analysis
    avg_daily_sales = brand_df['Daily Sales'].mean()
    last_month_avg_sales = brand_df['Daily Sales'].tail(30).mean()
    if last_month_avg_sales > avg_daily_sales:
        insights.append(f"The sales for {selected_brand} have shown an upward trend in the last 30 days.")
    else:
        insights.append(f"The sales for {selected_brand} have been relatively stable in the last 30 days.")

    # Identifying significant changes or events (you can customize this based on your data)
    # For example, if there's a significant increase or decrease in sales compared to historical data, you can mention it here.

    # Adding a general insight
    insights.append("The forecast for the next year shows potential growth in sales.")

    return insights


# Load the data
url = "https://raw.githubusercontent.com/sarakadd/SCOPSIS/main/REGIE%20PoC%20Data%20(1).csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content
# Read the data into a pandas DataFrame using the read_csv function with the URL directly
df = pd.read_csv(io.StringIO(download.decode('utf-8')))
# Extract relevant columns
df1 = df[['Date', 'Daily Sales', 'Tobacco_Company', 'Consumable_Brand', 'USD/LBP Rate', 'Price']]

# Filter the data for PM company and the specified products
pm_products = ['MARLBORO', 'HEETS', 'FIIT', 'BOND', 'MERIT', 'CHESTERFIELD']
filtered_df = df1[(df1['Tobacco_Company'] == 'PM') & (df1['Consumable_Brand'].isin(pm_products))]

# Ensure the 'Date' column is in a proper datetime format
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

# Create a Streamlit app
st.title('PM Demand Forecast')
selected_brand = st.selectbox('Select a brand', pm_products)

# Get data for the selected brand
brand_df = filtered_df[filtered_df['Consumable_Brand'] == selected_brand].copy()

# Perform SARIMA forecast and get the plot and MAPE
try:
    forecast_plot, mape, forecast_dates, forecast = sarima_forecast(brand_df)

    # Display the forecast plot using Plotly in Streamlit
    st.plotly_chart(forecast_plot, use_container_width=True)

    # Generate insights
    insights = generate_insights(brand_df, forecast_dates, forecast)

    # Display autogenerated insights
    st.markdown("<h3 style='font-size: 16px;'>Insights</h3>", unsafe_allow_html=True)
    for insight in insights:
        st.write(insight)
    # Display the MAPE value below the insights
    st.write(f"Mean Absolute Percentage Error for {selected_brand}: {mape:.2f}%")

except ValueError as e:
    st.error(f"Error occurred during the forecast for {selected_brand}: {e}")
