import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st

# Definition of constants
SITE = 'cgm-monitor-alfontal.herokuapp.com'
MAX_VALUES = 2000
LINEPLOT_HOURS = 4
TARGET_SEVERE_LOW = 50
TARGET_LOW = 70
TARGET_MILD_HIGH = 150
TARGET_HIGH = 180
TARGET_SEVERE_HIGH = 250
DIRECTIONS = {
    'DoubleDown': '⇊',
    'SingleDown': '↓',
    'FortyFiveDown': '↘',
    'Flat': '→',
    'FortyFiveUp': '↗',
    'SingleUp': '↑',
    'DoubleUp': '⇈'
}

# Cache directory
CACHE_DIR = Path(__file__).parent / '.cache'
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / 'cgm_data.pkl'
CACHE_DURATION = 92400  # 24 hours in seconds

# color_scheme
strong_red = '#960200'
light_red = '#CE6C47'
mild_yellow = '#FFD046'
light_green = '#49D49D'

def load_cached_data():
    """Load cached data if it exists and is recent enough."""
    if CACHE_FILE.exists():
        cache_age = datetime.datetime.now().timestamp() - CACHE_FILE.stat().st_mtime
        if cache_age < CACHE_DURATION:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    return None

def save_cached_data(data):
    """Save data to cache file."""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)

# Site Definition - Move to top before data fetching
st.set_page_config('SugarBoard 📈', layout='wide')

# Check if cached data exists
cached_data = load_cached_data()

# Create a placeholder for the loading callout at the top of the main area
loading_placeholder = st.empty()

# User choice: cached vs fresh data
st.sidebar.markdown("### Data Source")
if cached_data:
    cache_time = datetime.datetime.fromtimestamp(CACHE_FILE.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.info(f"💾 **Cached data available**\nLast updated: {cache_time}")
    use_cache = st.sidebar.radio(
        "Choose data source:",
        options=["Use cached data (fast)", "Fetch fresh data (slow)"],
        index=0
    ) == "Use cached data (fast)"
else:
    st.sidebar.warning("⚠️ No cached data available")
    use_cache = False
    st.sidebar.info("Fresh data will be fetched from Nightscout")

st.sidebar.markdown("---")

# Show persistent loading callout
if use_cache and cached_data:
    loading_placeholder.info("⏳ Loading cached data...")
    last_value = cached_data['last_value']
    previous_value = cached_data['previous_value']
    df_3months = cached_data['df_3months']  # The big 3-month dataset
    loading_placeholder.success("✅ Cached data loaded successfully!")
else:
    # Nightscout API calls and data processing
    loading_placeholder.warning("⏳ Fetching fresh data from Nightscout... This may take a minute.")
    st.sidebar.info("📡 Fetching fresh data from Nightscout...")
    
    try:
        # Get last 2 values first (small request)
        last_value, previous_value = requests.get(f'https://{SITE}/api/v1/entries.json?count=2', timeout=10).json()
    except Exception as e:
        st.error(f"⚠️ Failed to fetch latest values: {str(e)}")
        st.stop()
    
    # Fetch 90 days of data in chunks to avoid overwhelming the server
    CHUNK_SIZE = 10000  # Larger chunks with proper delays
    DAYS_TO_FETCH = 90
    REQUEST_TIMEOUT = 120  # Give server more time to respond
    DELAY_BETWEEN_REQUESTS = 3  # Wait 3 seconds between requests
    
    bg_categories = [
        f'<{TARGET_SEVERE_LOW}', 
        f'{TARGET_SEVERE_LOW}-{TARGET_LOW - 1}',
        f'{TARGET_LOW}-{TARGET_MILD_HIGH}',
        f'{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}',
        f'{TARGET_HIGH + 1}-{TARGET_SEVERE_HIGH}',
        f'>{TARGET_SEVERE_HIGH}']

    # Fetch data in chunks to be gentle on the server
    st.sidebar.info(f"📥 Fetching {DAYS_TO_FETCH} days of data with {DELAY_BETWEEN_REQUESTS}s delays between requests...")
    all_data = []
    
    import time  # Import at the top of the try block
    
    try:
        # Start from now and work backwards
        chunk_end = datetime.datetime.now()
        chunk_start = chunk_end - datetime.timedelta(days=DAYS_TO_FETCH)
        
        # Calculate number of chunks needed
        # Each chunk gets CHUNK_SIZE records, estimate ~1440 records per day (one every min)
        estimated_records = DAYS_TO_FETCH * 1440
        num_api_chunks = (estimated_records + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Fetch using count-based approach going backwards in time
        oldest_date = None
        for i in range(num_api_chunks):
            status_text.text(f"Fetching chunk {i+1}/{num_api_chunks}...")
            
            if oldest_date:
                # Continue from where we left off
                chunk_url = f'https://{SITE}/api/v1/entries/sgv.json?&find[dateString][$lt]={oldest_date.isoformat()}&count={CHUNK_SIZE}'
            else:
                # First chunk - get most recent data
                chunk_url = f'https://{SITE}/api/v1/entries/sgv.json?count={CHUNK_SIZE}'
            
            # Add delay BEFORE each request (except the first one) to give server time to cool down
            if i > 0:
                time.sleep(DELAY_BETWEEN_REQUESTS)
            
            response = requests.get(chunk_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            chunk_data = response.json()
            
            if not chunk_data:
                # No more data
                break
            
            all_data.extend(chunk_data)
            
            # Update oldest date for next iteration (make timezone-naive for comparison)
            oldest_date = pd.to_datetime(chunk_data[-1]['dateString'])
            if oldest_date.tzinfo is not None:
                oldest_date = oldest_date.tz_localize(None)  # Remove timezone info
            
            status_text.text(f"✓ Chunk {i+1}: {len(chunk_data)} records (oldest: {oldest_date.date()})")
            progress_bar.progress((i + 1) / num_api_chunks)
            
            # Check if we've reached our target date range
            if oldest_date < chunk_start:
                st.sidebar.info(f"✓ Reached target date {chunk_start.date()}")
                break
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            st.error("No data returned from the API. Please check your Nightscout site configuration.")
            st.stop()
        
        st.sidebar.success(f"✅ Fetched {len(all_data)} total records")
            
        # Process all fetched data into 3-month dataset
        df_raw = pd.DataFrame(all_data)
        df_raw['date'] = pd.to_datetime(df_raw['dateString'])
        df_raw = df_raw[['date', 'sgv', 'device']]
        df_raw = df_raw.drop_duplicates().set_index('date').sort_index()
        
        df_3months = (df_raw
              .resample('5 min')
              ['sgv']
              .mean()
              .reset_index()
              .assign(cat_glucose=lambda dd: pd.cut(dd['sgv'],
                                                    bins=[0, 50, 70, 150, 180, 250, np.inf],
                                                    labels=bg_categories))
              )
        
    except requests.exceptions.JSONDecodeError:
        loading_placeholder.error("⚠️ **Server Error**: The Nightscout server couldn't handle the request. This might be due to:\n"
                 "- **Memory issues on Heroku eco dyno** (most common)\n"
                 "- Rate limiting (too many requests)\n"
                 "- Server maintenance or dyno sleeping\n\n"
                 "**Solution**: The app now uses smaller chunks (500 records). Please wait 2-3 minutes and try again. "
                 "The server needs time to cool down.")
        st.stop()
    except requests.exceptions.RequestException as e:
        loading_placeholder.error(f"⚠️ Failed to fetch data from Nightscout API: {str(e)}\n\n"
                                  "If this is a timeout or connection error, the Heroku dyno may be sleeping or overloaded.")
        st.stop()

    # Cache the BIG 3-month data for future use
    save_cached_data({
        'last_value': last_value,
        'previous_value': previous_value,
        'df_3months': df_3months
    })
    st.sidebar.success("✅ Data cached! Next time select 'Use cached data' to skip server requests.")
    
    # Show dismissible success message
    with loading_placeholder.container():
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            st.success("✅ Fresh data fetched and cached successfully!")
        with col_btn:
            if st.button("Dismiss", key="dismiss_fetch_success"):
                loading_placeholder.empty()

# For cached data, show a dismissible info message
if use_cache and cached_data:
    # Replace the loading message with a dismissible success message
    with loading_placeholder.container():
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            st.info(f"ℹ️ Using cached data from {datetime.datetime.fromtimestamp(CACHE_FILE.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        with col_btn:
            if st.button("Dismiss", key="dismiss_cache_info"):
                loading_placeholder.empty()

# Continue with the rest of the app using the cached 3-month data
curr_dir = DIRECTIONS[last_value['direction']]
bg_categories = [
    f'<{TARGET_SEVERE_LOW}', 
    f'{TARGET_SEVERE_LOW}-{TARGET_LOW - 1}',
    f'{TARGET_LOW}-{TARGET_MILD_HIGH}',
    f'{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}',
    f'{TARGET_HIGH + 1}-{TARGET_SEVERE_HIGH}',
    f'>{TARGET_SEVERE_HIGH}']

st.sidebar.info(f"📊 3-month cache: {len(df_3months)} records from {df_3months.date.min().date()} to {df_3months.date.max().date()}")

# UI Layout
col1, col2 = st.columns([3, 2])
with col1:
    st.title('SugarBoard 📈')
    st.markdown('##### Bringing you closer to your own continuous glucose monitoring data.')
with col2:
    st.header(f'Last value: **{last_value["sgv"]}** mg/dl {curr_dir}')

st.markdown('---')

# Time in Range selector
tir_period = st.selectbox(
    'Select time period for Time in Range',
    options=['Last Day', 'Last Week', 'Last Month', 'Last 3 Months'],
    index=1  # Default to 'Last Week'
)

# Map selection to days
period_days = {
    'Last Day': 1,
    'Last Week': 7,
    'Last Month': 30,
    'Last 3 Months': 90
}

# Filter data based on selection
selected_days = period_days[tir_period]
selected_df = df_3months.loc[df_3months.date > df_3months.date.max() - pd.to_timedelta(f'{selected_days} days')]

# Calculate TIR data for selected period
tir_counts = selected_df['cat_glucose'].value_counts(normalize=True)
tir_data = pd.DataFrame({
    'cat_glucose': bg_categories,
    'value': [tir_counts.get(cat, 0) for cat in bg_categories],
    'percent_label': [(tir_counts.get(cat, 0) * 100).round(0).astype(int).astype(str) + '%' for cat in bg_categories]
})

# Get last 4 hours for recent chart
curr_df = df_3months.loc[df_3months.date > df_3months.date.max() - pd.to_timedelta('4 h')].copy()

# Get date range for glucose patterns (user selectable)
col3, col4 = st.columns([1, 1])
with col3:
    sel_initial_date = st.date_input('Initial date', value=datetime.datetime.now() - datetime.timedelta(days=90))
with col4:
    sel_final_date = st.date_input('Final date', value=datetime.datetime.now())

# Convert dates to datetime for comparison
sel_initial_dt = pd.to_datetime(sel_initial_date)
sel_final_dt = pd.to_datetime(sel_final_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of day

# Match timezone info with df_3months.date
if df_3months.date.dt.tz is not None:
    # df_3months has timezone info, localize our comparison dates to the same timezone
    if sel_initial_dt.tzinfo is None:
        sel_initial_dt = sel_initial_dt.tz_localize(df_3months.date.dt.tz)
        sel_final_dt = sel_final_dt.tz_localize(df_3months.date.dt.tz)
    else:
        sel_initial_dt = sel_initial_dt.tz_convert(df_3months.date.dt.tz)
        sel_final_dt = sel_final_dt.tz_convert(df_3months.date.dt.tz)
else:
    # df_3months is timezone-naive, ensure our dates are too
    if sel_initial_dt.tzinfo is not None:
        sel_initial_dt = sel_initial_dt.tz_localize(None)
        sel_final_dt = sel_final_dt.tz_localize(None)

# Filter 3-month data for selected pattern range
sel_df = df_3months[
    (df_3months.date >= sel_initial_dt) & 
    (df_3months.date <= sel_final_dt)
].copy()

# Time in Range Chart - Single plot without facets
bg_colors = {
    f'<{TARGET_SEVERE_LOW}': strong_red,
    f'{TARGET_SEVERE_LOW}-{TARGET_LOW - 1}': light_red,
    f'{TARGET_LOW}-{TARGET_MILD_HIGH}': light_green,
    f'{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}': mild_yellow,
    f'{TARGET_HIGH + 1}-{TARGET_SEVERE_HIGH}': light_red,
    f'>{TARGET_SEVERE_HIGH}': strong_red
}

fig_tir = px.bar(
    tir_data,
    x='cat_glucose',
    y='value',
    color='cat_glucose',
    text='percent_label',
    category_orders={'cat_glucose': bg_categories},
    color_discrete_map=bg_colors,
    title=f'Time in Range - {tir_period}'
)

fig_tir.update_traces(
    textposition='outside',
    cliponaxis=False,
    hovertemplate='Range: %{x}<br>Percentage: %{y:.0%}<extra></extra>',
    width=0.8
)

fig_tir.update_yaxes(
    tickformat='.0%',
    title=None,
    range=[0, tir_data['value'].max() * 1.15]  # Add space for labels
)

fig_tir.update_xaxes(
    title=None,
    tickangle=-45
)

fig_tir.update_layout(
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=10),
    margin=dict(t=60, r=20, b=80, l=40),
    height=400
)

# Recent CGM values chart (Interactive Plotly)
# Handle timezone conversion - check if already timezone-aware
if curr_df['date'].dt.tz is None:
    curr_df['date_local'] = curr_df['date'].dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid')
else:
    curr_df['date_local'] = curr_df['date'].dt.tz_convert('Europe/Madrid')

fig_recent = px.line(
    curr_df,
    x='date_local',
    y='sgv',
    color='device' if 'device' in curr_df.columns and curr_df.device.nunique() > 1 else None,
    markers=True,
    hover_data={'date_local': '|%H:%M', 'sgv': True}
)

fig_recent.update_traces(
    mode='lines+markers',
    line=dict(width=2),
    marker=dict(size=6)
)

fig_recent.add_hline(
    y=TARGET_LOW,
    line_dash='dash',
    line_color=light_green,
    opacity=0.7,
    annotation_text='Low',
    annotation_position='right'
)

fig_recent.add_hline(
    y=TARGET_MILD_HIGH,
    line_dash='dash',
    line_color=light_green,
    opacity=0.7,
    annotation_text='Mild High',
    annotation_position='right'
)

fig_recent.add_hline(
    y=TARGET_HIGH,
    line_dash='dash',
    line_color=mild_yellow,
    opacity=0.7,
    annotation_text='High',
    annotation_position='right'
)

fig_recent.update_xaxes(title=None)
fig_recent.update_yaxes(title='Glucose [mg/dl]')

fig_recent.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        title_text=None  # No device legend needed
    ),
    hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=10),
    margin=dict(t=60, r=20, b=40, l=60),
    height=400
)

# Glucose Patterns Chart - using filtered 3-month data
# Debug: Show data info
st.sidebar.info(f"📊 Pattern data: {len(sel_df)} records from {sel_df.date.min().date()} to {sel_df.date.max().date()}")
st.sidebar.info(f"📊 Non-null SGV values: {sel_df.sgv.notna().sum()}")

sel_df_quantiles = (sel_df
                     .dropna(subset=['sgv'])  # Remove any NaN values first
                     .assign(hour=lambda dd: dd.date.dt.hour + dd.date.dt.minute / 60)
                     .groupby('hour', as_index=False)
                     .agg({
                         'sgv': [
                             ('median', 'median'),
                             ('q90', lambda x: x.quantile(.9)),
                             ('q10', lambda x: x.quantile(.1)),
                             ('q25', lambda x: x.quantile(.25)),
                             ('q75', lambda x: x.quantile(.75))
                         ]
                     })
)

# Flatten the column names from MultiIndex
sel_df_quantiles.columns = ['hour', 'median', 'q90', 'q10', 'q25', 'q75']

st.sidebar.info(f"📊 Quantile data points: {len(sel_df_quantiles)}")

# Apply smoothing only if we have enough data points
if len(sel_df_quantiles) >= 10:
    sel_df_quantiles[['median', 'q90', 'q10', 'q25', 'q75']] = (
        sel_df_quantiles[['median', 'q90', 'q10', 'q25', 'q75']]
        .rolling(window=10, center=True, min_periods=1)
        .mean()
    )

# Format hour as HH:MM for display
sel_df_quantiles['hour_formatted'] = sel_df_quantiles['hour'].apply(
    lambda h: f"{int(h):02d}:{int((h % 1) * 60):02d}"
)

# Glucose patterns chart (Interactive Plotly) - Modern look with gradient
fig_patterns = go.Figure()

# Add 10th-90th percentile ribbon with gradient blue
fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q90'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip',
    customdata=sel_df_quantiles[['hour_formatted']]
))

fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q10'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(99, 102, 241, 0.15)',
    line=dict(color='rgba(99, 102, 241, 0.3)', width=1),
    name='10-90th percentile',
    hovertemplate='<b>10-90th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>',
    customdata=sel_df_quantiles[['hour_formatted', 'q10', 'q90']].values
))

# Add 25th-75th percentile ribbon with deeper blue
fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q75'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip',
    customdata=sel_df_quantiles[['hour_formatted']]
))

fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q25'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(99, 102, 241, 0.35)',
    line=dict(color='rgba(99, 102, 241, 0.6)', width=1),
    name='25-75th percentile',
    hovertemplate='<b>25-75th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>',
    customdata=sel_df_quantiles[['hour_formatted', 'q25', 'q75']].values
))

# Add median line with glow effect
fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['median'],
    mode='lines',
    line=dict(color='#22D3EE', width=3, shape='spline'),
    name='Median',
    hovertemplate='<b>Median:</b> %{y:.0f} mg/dl<extra></extra>',
    customdata=sel_df_quantiles[['hour_formatted']].values
))

# Add target range lines with annotations
fig_patterns.add_hline(
    y=TARGET_LOW,
    line_dash='dash',
    line_color='#34D399',
    opacity=0.8,
    line_width=2,
    annotation_text='Target Low',
    annotation_position='right',
    annotation=dict(font_size=9, font_color='#34D399')
)

fig_patterns.add_hline(
    y=TARGET_MILD_HIGH,
    line_dash='dash',
    line_color='#34D399',
    opacity=0.8,
    line_width=2,
    annotation_text='Target Mild High',
    annotation_position='right',
    annotation=dict(font_size=9, font_color='#34D399')
)

fig_patterns.add_hline(
    y=TARGET_HIGH,
    line_dash='dash',
    line_color='#FBBF24',
    opacity=0.8,
    line_width=2,
    annotation_text='High',
    annotation_position='right',
    annotation=dict(font_size=9, font_color='#FBBF24')
)

fig_patterns.update_xaxes(
    title=None,
    tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
    ticktext=['0h', '3h', '6h', '9h', '12h', '15h', '18h', '21h'],
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(255, 255, 255, 0.1)'
)

fig_patterns.update_yaxes(
    title='Glucose [mg/dl]',
    range=[40, None],
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(255, 255, 255, 0.1)'
)

fig_patterns.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        bgcolor='rgba(0,0,0,0.3)',
        bordercolor='rgba(99, 102, 241, 0.5)',
        borderwidth=1
    ),
    font=dict(color='white', size=10),
    margin=dict(t=60, r=20, b=40, l=60),
    height=400,
    hovermode='x unified',
    xaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=False,
        showgrid=True,
        hoverformat='%H:%M'  # Format x-axis values in hover as HH:MM
    ),
    hoverlabel=dict(
        bgcolor='rgba(17, 24, 39, 0.95)',
        font_size=11,
        font_family='Arial',
        font_color='white'
    )
)

# Update traces to remove redundant time from individual hovers
for trace in fig_patterns.data:
    if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
        # Remove formatted time from each trace since it's now in the unified title
        if trace.name == '10-90th percentile':
            trace.hovertemplate = '<b>10-90th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>'
        elif trace.name == '25-75th percentile':
            trace.hovertemplate = '<b>25-75th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>'
        elif trace.name == 'Median':
            trace.hovertemplate = '<b>Median:</b> %{y:.0f} mg/dl<extra></extra>'

st.markdown('---')

c1, c2, c3 = st.columns([3, 4, 4])
with c1:
    st.markdown('#### Time in range (%) during last periods')
    st.plotly_chart(fig_tir, use_container_width=True)

with c2:
    st.markdown('#### CGM values from the last 4 hours')
    st.plotly_chart(fig_recent, use_container_width=True)

with c3:
    st.markdown('#### Glucose patterns from selected period')
    st.plotly_chart(fig_patterns, use_container_width=True)
