import datetime
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.ui_theme import (
    PLOT_CONFIG,
    THEME_BORDER,
    THEME_MUTED,
    THEME_SURFACE,
    THEME_TEXT,
    inject_base_theme,
    panel_card,
)

# Definition of constants
SITE = 'cgm-monitor-alfontal.herokuapp.com'
MAX_VALUES = 2000
LINEPLOT_HOURS = 4
RECENT_POINTS = LINEPLOT_HOURS * 75  # buffer above one reading per minute
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
RECENT_CACHE_FILE = CACHE_DIR / 'recent_data.pkl'
RECENT_CACHE_DURATION = 300  # 5 minutes
RECENT_REQUEST_TIMEOUT = 20


def fetch_recent_data():
    """Fetch the most recent CGM entries and prep a 4-hour dataframe."""
    response = requests.get(
        f'https://{SITE}/api/v1/entries/sgv.json',
        params={'count': RECENT_POINTS},
        timeout=RECENT_REQUEST_TIMEOUT
    )
    response.raise_for_status()
    entries = response.json()
    if not entries:
        raise ValueError('Nightscout returned no recent entries.')

    df_recent = pd.DataFrame(entries)
    if 'dateString' not in df_recent:
        raise ValueError('Recent data payload missing dateString field.')

    df_recent['date'] = pd.to_datetime(df_recent['dateString'], utc=True)
    df_recent = (df_recent
                 [['date', 'sgv', 'device']]
                 .drop_duplicates()
                 .set_index('date')
                 .sort_index())

    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=LINEPLOT_HOURS)
    df_recent = df_recent.loc[df_recent.index >= cutoff]
    df_recent = df_recent.reset_index()

    last_value = entries[0]
    previous_value = entries[1] if len(entries) > 1 else entries[0]

    return last_value, previous_value, df_recent

# Color palettes
strong_red = '#960200'
light_red = '#CE6C47'
mild_yellow = '#FFD046'
light_green = '#49D49D'

def _load_cache(path: Path, max_age_seconds: int):
    """Helper to load pickle cache with expiration."""
    if path.exists():
        cache_age = time.time() - path.stat().st_mtime
        if cache_age < max_age_seconds:
            with open(path, 'rb') as fh:
                return pickle.load(fh)
    return None


def _save_cache(path: Path, data):
    """Helper to persist pickle cache."""
    with open(path, 'wb') as fh:
        pickle.dump(data, fh)


def load_cached_data():
    """Load cached historical data."""
    return _load_cache(CACHE_FILE, CACHE_DURATION)


def save_cached_data(data):
    """Persist cached historical data."""
    _save_cache(CACHE_FILE, data)


def load_recent_cache():
    """Load the cached recent (last hours) data."""
    return _load_cache(RECENT_CACHE_FILE, RECENT_CACHE_DURATION)


def save_recent_cache(data):
    """Persist the cached recent (last hours) data."""
    _save_cache(RECENT_CACHE_FILE, data)


def fetch_latest_entry():
    """Fetch only the most recent CGM entry for minute-level refresh."""
    response = requests.get(
        f'https://{SITE}/api/v1/entries/sgv.json',
        params={'count': 1},
        timeout=RECENT_REQUEST_TIMEOUT
    )
    response.raise_for_status()
    entries = response.json()
    if not entries:
        raise ValueError('Nightscout returned no latest entry.')
    return entries[0]


def parse_entry_timestamp(entry):
    """Return a timezone-aware timestamp for a Nightscout entry."""
    ts_raw = entry.get('dateString') or entry.get('date')
    if ts_raw is None:
        return None
    return pd.to_datetime(ts_raw, utc=True)

# Site Definition - Move to top before data fetching
st.set_page_config('SugarBoard 📈', layout='wide')

inject_base_theme()

if 'minute_refresh_enabled' not in st.session_state:
    st.session_state['minute_refresh_enabled'] = False

# Clear any pending auto-refresh timer immediately on rerun
st.markdown(
    """
    <script>
        if (window.sugarboardRefreshTimer) {
            clearTimeout(window.sugarboardRefreshTimer);
        }
    </script>
    """,
    unsafe_allow_html=True
)

# Check if cached data exists
cached_data = load_cached_data()
fallback_last_value = None
fallback_previous_value = None
df_3months = None

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
    df_3months = cached_data['df_3months']  # The big 3-month dataset
    fallback_last_value = cached_data.get('last_value')
    fallback_previous_value = cached_data.get('previous_value')
    loading_placeholder.success("✅ Cached data loaded successfully!")
    st.session_state['minute_refresh_enabled'] = True
    loading_placeholder.empty()
else:
    # Nightscout API calls and data processing
    st.session_state['minute_refresh_enabled'] = False
    loading_placeholder.warning("⏳ Fetching fresh data from Nightscout... This may take a minute.")
    st.sidebar.info("📡 Fetching fresh data from Nightscout...")
    
    try:
        # Get last 2 values first (small request) for fallback purposes
        latest_entries = requests.get(f'https://{SITE}/api/v1/entries.json?count=2', timeout=10).json()
        if not latest_entries:
            raise ValueError("Nightscout did not return any recent entries.")
        fallback_last_value = latest_entries[0]
        fallback_previous_value = latest_entries[1] if len(latest_entries) > 1 else latest_entries[0]
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
        loading_placeholder.error(
            "⚠️ **Server Error**: The Nightscout server couldn't handle the request. This might be due to:\n"
            "- **Memory pressure on the Heroku eco dyno**\n"
            "- Rate limiting (too many requests)\n"
            "- Server maintenance or dyno sleeping\n\n"
            "**What to do**: Please wait 2-3 minutes before retrying so the dyno can recover. "
            "We now fetch in larger 10k batches with enforced cooldowns, but the server may still need a breather."
        )
        st.stop()
    except requests.exceptions.RequestException as e:
        loading_placeholder.error(f"⚠️ Failed to fetch data from Nightscout API: {str(e)}\n\n"
                                  "If this is a timeout or connection error, the Heroku dyno may be sleeping or overloaded.")
        st.stop()

    # Cache the BIG 3-month data for future use
    save_cached_data({
        'last_value': fallback_last_value,
        'previous_value': fallback_previous_value,
        'df_3months': df_3months
    })
    st.session_state['minute_refresh_enabled'] = True
    st.sidebar.success("✅ Data cached! Next time select 'Use cached data' to skip server requests.")
    
    # Show dismissible success message
    with loading_placeholder.container():
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            st.success("✅ Fresh data fetched and cached successfully!")
        with col_btn:
            if st.button("Dismiss", key="dismiss_fetch_success"):
                loading_placeholder.empty()

# Ensure we have historical data loaded before proceeding
if df_3months is None:
    st.error("Historical dataset not available. Please refresh the app to try again.")
    st.stop()

assert df_3months is not None

# Fetch or load recent data for current metrics and last-hours chart
recent_cache = load_recent_cache()
recent_data_source = 'cache'
recent_fetched_ts = None

if recent_cache:
    last_value = recent_cache['last_value']
    previous_value = recent_cache['previous_value']
    df_recent = recent_cache['df_recent']
    recent_fetched_ts = recent_cache.get('fetched_at')
else:
    try:
        last_value, previous_value, df_recent = fetch_recent_data()
        recent_fetched_ts = time.time()
        save_recent_cache({
            'last_value': last_value,
            'previous_value': previous_value,
            'df_recent': df_recent,
            'fetched_at': recent_fetched_ts
        })
        recent_data_source = 'api'
    except Exception as recent_err:
        if fallback_last_value is not None:
            st.warning(
                f"⚠️ Could not refresh recent data ({recent_err}). Displaying cached fallback values."
            )
            last_value = fallback_last_value
            previous_value = fallback_previous_value or fallback_last_value
            df_recent = df_3months.loc[
                df_3months.date > df_3months.date.max() - pd.to_timedelta(f'{LINEPLOT_HOURS} h')
            ].copy()
            if 'device' not in df_recent.columns:
                df_recent['device'] = 'Unknown'
            recent_fetched_ts = CACHE_FILE.stat().st_mtime if CACHE_FILE.exists() else None
            recent_data_source = 'fallback'
            st.session_state['minute_refresh_enabled'] = True
        else:
            st.error(f"⚠️ Unable to load recent glucose data: {recent_err}")
            st.stop()

# Ensure recent dataframe uses timezone-aware timestamps for comparisons
if not df_recent.empty and not is_datetime64_any_dtype(df_recent['date']):
    df_recent['date'] = pd.to_datetime(df_recent['date'], utc=True)

# Minute-level refresh: fetch only the latest entry and update the cache if newer
try:
    latest_entry = fetch_latest_entry()
    latest_ts = parse_entry_timestamp(latest_entry)
    cached_ts = parse_entry_timestamp(last_value)

    if latest_ts is not None and (cached_ts is None or latest_ts > cached_ts):
        previous_value = last_value
        last_value = latest_entry

        latest_row = pd.DataFrame([
            {
                'date': latest_ts,
                'sgv': latest_entry.get('sgv'),
                'device': latest_entry.get('device', 'Unknown') or 'Unknown'
            }
        ])

        df_recent = pd.concat([df_recent, latest_row], ignore_index=True)
        df_recent = (
            df_recent
            .drop_duplicates(subset='date', keep='last')
            .sort_values('date')
        )

        recent_cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=LINEPLOT_HOURS)
        df_recent = df_recent.loc[df_recent['date'] >= recent_cutoff].reset_index(drop=True)

        recent_fetched_ts = time.time()
        save_recent_cache({
            'last_value': last_value,
            'previous_value': previous_value,
            'df_recent': df_recent,
            'fetched_at': recent_fetched_ts
        })
        recent_data_source = 'api'
except Exception as latest_err:
    st.sidebar.warning(f"⚠️ Live reading refresh failed: {latest_err}")

# Continue with the rest of the app using the datasets
trend_raw = last_value.get('direction') or 'Flat'
curr_dir = DIRECTIONS.get(trend_raw, '→')
bg_categories = [
    f'<{TARGET_SEVERE_LOW}', 
    f'{TARGET_SEVERE_LOW}-{TARGET_LOW - 1}',
    f'{TARGET_LOW}-{TARGET_MILD_HIGH}',
    f'{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}',
    f'{TARGET_HIGH + 1}-{TARGET_SEVERE_HIGH}',
    f'>{TARGET_SEVERE_HIGH}']

st.sidebar.markdown(
    (
        f"<div class='sidebar-card'>"
        f"<strong>Cache Status</strong>"
        f"<span>{len(df_3months):,} records</span><br/>"
        f"<span>{df_3months.date.min().date()} → {df_3months.date.max().date()}</span>"
        f"</div>"
    ),
    unsafe_allow_html=True
)

recent_timestamp_label = (
    datetime.datetime.fromtimestamp(recent_fetched_ts).strftime('%H:%M:%S')
    if recent_fetched_ts else 'unknown'
)
recent_source_label = {
    'cache': 'Cached (≤5 min)',
    'api': 'Live refresh',
    'fallback': 'Fallback'
}.get(recent_data_source, recent_data_source)

st.sidebar.markdown(
    (
        f"<div class='sidebar-card'>"
        f"<strong>Recent Data</strong>"
        f"<span>Source: {recent_source_label}</span><br/>"
        f"<span>Fetched at: {recent_timestamp_label}</span>"
        f"</div>"
    ),
    unsafe_allow_html=True
)

# Hero header with current status
last_timestamp_raw = last_value.get('dateString') or last_value.get('date')
last_dt_utc = pd.to_datetime(last_timestamp_raw, utc=True) if last_timestamp_raw is not None else pd.Timestamp.utcnow()
last_dt_local = last_dt_utc.tz_convert('Europe/Madrid')
delta_value = last_value['sgv'] - previous_value['sgv']
delta_class = 'positive' if delta_value > 0 else 'negative' if delta_value < 0 else 'neutral'
delta_text = f"{delta_value:+.0f} mg/dL"
trend_label = trend_raw.replace('FortyFive', '45')
device_label = last_value.get('device') or '—'

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-meta">
            <h1>SugarBoard</h1>
            <p>Realtime CGM insights.</p>
        </div>
        <div class="hero-badges">
            <div class="hero-badge">
                <span class="label">Last Reading</span>
                <span class="value">{last_value['sgv']} mg/dL {curr_dir}</span>
            </div>
            <div class="hero-badge">
                <span class="label">Change vs prev</span>
                <span class="value {delta_class}">{delta_text}</span>
            </div>
            <div class="hero-badge">
                <span class="label">Updated</span>
                <span class="value">{last_dt_local.strftime('%d %b %Y · %H:%M')}</span>
            </div>
            <div class="hero-badge">
                <span class="label">Device</span>
                <span class="value">{device_label}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Controls for charts
default_initial_date = (datetime.datetime.now() - datetime.timedelta(days=90)).date()
default_final_date = datetime.datetime.now().date()

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

metric_cols = st.columns((1.1, 1, 1, 1))

with metric_cols[0]:
    st.markdown("<div class='metric-card metric-card--form'>", unsafe_allow_html=True)
    st.markdown("<span class='metric-label'>Time in Range window</span>", unsafe_allow_html=True)
    tir_period = st.selectbox(
        'Time in Range window',
        options=['Last Day', 'Last Week', 'Last Month', 'Last 3 Months'],
        index=1,
        key='tir_period',
        label_visibility='collapsed'
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
tir_value_max = max(tir_data['value'].max(), 0.0001)

# Metrics overview cards
tir_core_pct = tir_counts.get(f'{TARGET_LOW}-{TARGET_MILD_HIGH}', 0) * 100
tir_extended_pct = tir_counts.get(f'{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}', 0) * 100
tir_in_range_pct = tir_core_pct + tir_extended_pct
average_glucose = selected_df['sgv'].mean()
hypo_events = int((selected_df['sgv'] < TARGET_LOW).sum())
records_selected = len(selected_df)

if np.isnan(average_glucose):
    avg_display = "--"
    mmol_display = "--"
else:
    avg_display = f"{average_glucose:.0f} mg/dL"
    mmol_display = f"{average_glucose * 0.0555:.1f}"

tir_breakdown_html = (
    f"<span class='tir-breakdown'>"
    f"<span class='tir-chip' style='color:{light_green};' title='70-150 mg/dL'>{tir_core_pct:.0f}%</span>"
    f"<span class='tir-chip' style='color:{mild_yellow};' title='151-180 mg/dL'>{tir_extended_pct:.0f}%</span>"
    "</span>"
)

with metric_cols[1]:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Time In Range</span>
            <span class="metric-value">{tir_in_range_pct:.0f}% {tir_breakdown_html}</span>
            <span class="metric-caption">Window · {tir_period}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Average Glucose</span>
            <span class="metric-value">{avg_display}</span>
            <span class="metric-caption">{mmol_display} mmol/L equivalent</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Dataset</span>
            <span class="metric-value">{records_selected:,}</span>
            <span class="metric-caption">Hypo events: {hypo_events}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

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
    color_discrete_map=bg_colors
)

fig_tir.update_traces(
    textposition='outside',
    cliponaxis=False,
    hovertemplate='Range: %{x}<br>Percentage: %{y:.0%}<extra></extra>',
    width=0.95
)

fig_tir.update_yaxes(
    tickformat='.0%',
    title=None,
    range=[0, tir_value_max * 1.15],  # Add space for labels
    color=THEME_MUTED,
    gridcolor='rgba(93, 220, 255, 0.05)'
)

fig_tir.update_xaxes(
    title=None,
    tickangle=-45,
    color=THEME_MUTED,
    gridcolor='rgba(93, 220, 255, 0.05)'
)

fig_tir.update_layout(
    showlegend=False,
    paper_bgcolor=THEME_SURFACE,
    plot_bgcolor=THEME_SURFACE,
    font=dict(color=THEME_TEXT, size=12, family='Inter, sans-serif'),
    margin=dict(t=50, r=20, b=70, l=40),
    height=380
)

# Recent CGM values chart (Interactive Plotly)
# Handle timezone conversion - check if already timezone-aware
recent_chart_df = df_recent.copy()
if recent_chart_df['date'].dt.tz is None:
    recent_chart_df['date_local'] = recent_chart_df['date'].dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid')
else:
    recent_chart_df['date_local'] = recent_chart_df['date'].dt.tz_convert('Europe/Madrid')

fig_recent = px.line(
    recent_chart_df,
    x='date_local',
    y='sgv',
    color='device' if 'device' in recent_chart_df.columns and recent_chart_df.device.nunique() > 1 else None,
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

fig_recent.update_xaxes(
    title=None,
    color=THEME_MUTED,
    gridcolor='rgba(93, 220, 255, 0.05)'
)
fig_recent.update_yaxes(
    title='Glucose [mg/dl]',
    color=THEME_MUTED,
    gridcolor='rgba(93, 220, 255, 0.05)'
)

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
    paper_bgcolor=THEME_SURFACE,
    plot_bgcolor=THEME_SURFACE,
    font=dict(color=THEME_TEXT, size=12, family='Inter, sans-serif'),
    margin=dict(t=50, r=20, b=40, l=50),
    height=380
)

chart_col_left, chart_col_right = st.columns((1, 1))
with chart_col_left:
    with panel_card('Time in Range', tir_period):
        st.plotly_chart(fig_tir, width='stretch', config=PLOT_CONFIG)

with chart_col_right:
    with panel_card('Recent Glucose', 'Last 4 hours'):
        st.plotly_chart(fig_recent, width='stretch', config=PLOT_CONFIG)

with panel_card('Glucose Patterns', 'Customize window'):
    pattern_ctrl_cols = st.columns((1, 1))
    with pattern_ctrl_cols[0]:
        st.markdown("<span class='metric-label'>Patterns from</span>", unsafe_allow_html=True)
        sel_initial_date = st.date_input(
            'Patterns from',
            value=default_initial_date,
            max_value=default_final_date,
            key='patterns_start',
            label_visibility='collapsed'
        )
    with pattern_ctrl_cols[1]:
        st.markdown("<span class='metric-label'>Patterns to</span>", unsafe_allow_html=True)
        sel_final_date = st.date_input(
            'Patterns to',
            value=default_final_date,
            max_value=default_final_date,
            key='patterns_end',
            label_visibility='collapsed'
        )

    if sel_initial_date > sel_final_date:
        st.error("Initial date must be on or before the final date for glucose patterns.")
        st.stop()

    sel_initial_dt = pd.to_datetime(sel_initial_date)
    sel_final_dt = pd.to_datetime(sel_final_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if df_3months.date.dt.tz is not None:
        if sel_initial_dt.tzinfo is None:
            sel_initial_dt = sel_initial_dt.tz_localize(df_3months.date.dt.tz)
            sel_final_dt = sel_final_dt.tz_localize(df_3months.date.dt.tz)
        else:
            sel_initial_dt = sel_initial_dt.tz_convert(df_3months.date.dt.tz)
            sel_final_dt = sel_final_dt.tz_convert(df_3months.date.dt.tz)
    else:
        if sel_initial_dt.tzinfo is not None:
            sel_initial_dt = sel_initial_dt.tz_localize(None)
            sel_final_dt = sel_final_dt.tz_localize(None)

    sel_df = df_3months[
        (df_3months.date >= sel_initial_dt) &
        (df_3months.date <= sel_final_dt)
    ].copy()

    if sel_df.empty:
        pattern_window_text = "No data in range"
        valid_sgv_count = 0
        sel_df_quantiles = pd.DataFrame(columns=['hour', 'median', 'q90', 'q10', 'q25', 'q75'])
    else:
        pattern_window_text = f"{sel_df.date.min().date()} → {sel_df.date.max().date()}"
        valid_sgv_count = int(sel_df.sgv.notna().sum())
        sel_df_quantiles = (sel_df
                             .dropna(subset=['sgv'])
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
        sel_df_quantiles.columns = ['hour', 'median', 'q90', 'q10', 'q25', 'q75']

        if len(sel_df_quantiles) >= 10:
            sel_df_quantiles[['median', 'q90', 'q10', 'q25', 'q75']] = (
                sel_df_quantiles[['median', 'q90', 'q10', 'q25', 'q75']]
                .rolling(window=10, center=True, min_periods=1)
                .mean()
            )

        sel_df_quantiles['hour_formatted'] = sel_df_quantiles['hour'].apply(
            lambda h: f"{int(h):02d}:{int((h % 1) * 60):02d}"
        )

    st.sidebar.markdown(
        (
            f"<div class='sidebar-card'>"
            f"<strong>Pattern Dataset</strong>"
            f"<span>Records: {len(sel_df):,}</span><br/>"
            f"<span>Valid SGVs: {valid_sgv_count:,}</span><br/>"
            f"<span>Window: {pattern_window_text}</span>"
            f"</div>"
        ),
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        (
            f"<div class='sidebar-card'>"
            f"<strong>Quantile Breakdown</strong>"
            f"<span>Points: {len(sel_df_quantiles):,}</span>"
            f"</div>"
        ),
        unsafe_allow_html=True
    )

    if sel_df.empty or sel_df_quantiles.empty:
        st.info("No glucose records available for this window. Try expanding the date range.")
    else:
        fig_patterns = go.Figure()

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

        fig_patterns.add_trace(go.Scatter(
            x=sel_df_quantiles['hour'],
            y=sel_df_quantiles['median'],
            mode='lines',
            line=dict(color='#22D3EE', width=3, shape='spline'),
            name='Median',
            hovertemplate='<b>Median:</b> %{y:.0f} mg/dl<extra></extra>',
            customdata=sel_df_quantiles[['hour_formatted']].values
        ))

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
            gridcolor='rgba(93, 220, 255, 0.05)',
            color=THEME_MUTED
        )

        fig_patterns.update_yaxes(
            title='Glucose [mg/dl]',
            range=[40, None],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(93, 220, 255, 0.05)',
            color=THEME_MUTED
        )

        fig_patterns.update_layout(
            paper_bgcolor=THEME_SURFACE,
            plot_bgcolor=THEME_SURFACE,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(11, 18, 33, 0.85)',
                bordercolor=THEME_BORDER,
                borderwidth=1
            ),
            font=dict(color=THEME_TEXT, size=12, family='Inter, sans-serif'),
            margin=dict(t=60, r=30, b=40, l=60),
            height=400,
            hovermode='x unified',
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=False,
                showgrid=True,
                hoverformat='%H:%M'
            ),
            hoverlabel=dict(
                bgcolor='rgba(17, 24, 39, 0.95)',
                font_size=11,
                font_family='Inter, sans-serif',
                font_color=THEME_TEXT
            )
        )

        for trace in fig_patterns.data:
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                if trace.name == '10-90th percentile':
                    trace.hovertemplate = '<b>10-90th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>'
                elif trace.name == '25-75th percentile':
                    trace.hovertemplate = '<b>25-75th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>'
                elif trace.name == 'Median':
                    trace.hovertemplate = '<b>Median:</b> %{y:.0f} mg/dl<extra></extra>'

        st.markdown(
            f"<span class='metric-label' style='display:block;margin-bottom:0.5rem;'>Window: {pattern_window_text}</span>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_patterns, use_container_width=True, config=PLOT_CONFIG)

refresh_instruction = "window.sugarboardRefreshTimer = setTimeout(() => window.location.reload(), 60000);" if st.session_state.get('minute_refresh_enabled', False) else ""
st.markdown(
    f"""
    <script>
        if (window.sugarboardRefreshTimer) {{
            clearTimeout(window.sugarboardRefreshTimer);
        }}
        {refresh_instruction}
    </script>
    """,
    unsafe_allow_html=True
)
