import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st

# Definition of constants
SITE = 'cgm-monitor-alfontal.herokuapp.com'
MAX_VALUES = 999999
LINEPLOT_HOURS = 4
TARGET_LOW = 70
TARGET_HIGH = 150
DIRECTIONS = {
    'DoubleDown': '⇊',
    'SingleDown': '↓',
    'FortyFiveDown': '↘',
    'Flat': '→',
    'FortyFiveUp': '↗',
    'SingleUp': '↑',
    'DoubleUp': '⇈'
}

# Nightscout API calls and data processing
last_value, previous_value = requests.get(f'https://{SITE}/api/v1/entries.json?count=2').json()
curr_dir = DIRECTIONS[last_value['direction']]
prev_time = datetime.datetime.now() - datetime.timedelta(hours=LINEPLOT_HOURS)
initial_date = datetime.datetime.now() - datetime.timedelta(days=90)
final_date = datetime.datetime.now() + datetime.timedelta(days=1)
url = f'https://{SITE}/api/v1/entries/sgv.json?&find[dateString][$gte]={initial_date}' \
      f'&find[dateString][$lte]={final_date}&count={MAX_VALUES}'

current_url = f'https://{SITE}/api/v1/entries/sgv.json?&find[dateString][$gte]={prev_time}&count=100'

bg_categories = ['<50', '50-69', '70-150', '151-180', '181-250', '>250']

df = (pd.DataFrame(requests.get(url).json())
      .assign(date=lambda dd: pd.to_datetime(dd.dateString))
      [['date', 'sgv', 'device']]
      .set_index('date')
      .resample('5 min')
      ['sgv']
      .mean()
      .reset_index()
      .assign(cat_glucose=lambda dd: pd.cut(dd['sgv'],
                                            bins=[0, 50, 70, 150, 180, 250, np.inf],
                                            labels=bg_categories))
      )

last_24 = df.loc[df.date > df.date.max() - pd.to_timedelta('24 h')].assign(group='Last Day')
last_week = df.loc[df.date > df.date.max() - pd.to_timedelta('7 days')].assign(group='Last Week')
last_month = df.loc[df.date > df.date.max() - pd.to_timedelta('30 days')].assign(group='Last Month')
last_90 = df.loc[df.date > df.date.max() - pd.to_timedelta('90 days')].assign(group='Last 3 Months')
curr_df = (pd.DataFrame(requests.get(current_url.replace(' ', 'T')).json())
    .assign(date=lambda dd: pd.to_datetime(dd.dateString))
[['date', 'sgv', 'device']])

# Figures
tir_df = (pd.concat([last_24, last_week, last_month, last_90])
            .groupby('group')
            .apply(lambda dd: dd['cat_glucose'].value_counts(True)[bg_categories])
            .reset_index()
            .melt('group')
            .assign(percent_label=lambda dd: (dd['value'].round(2) * 100).astype(int).astype(str) + '%')
            .assign(group=lambda dd: pd.Categorical(dd.group, ordered=True,
                    categories=['Last Day', 'Last Week', 'Last Month', 'Last 3 Months']))
            .assign(cat_glucose=lambda dd: pd.Categorical(dd.cat_glucose, categories=bg_categories, ordered=True))
     )

# Time in Range Chart (Interactive Plotly)
bg_colors = {
    '<50': '#960200',
    '50-69': '#CE6C47',
    '70-150': '#49D49D',
    '151-180': '#FFD046',
    '181-250': '#CE6C47',
    '>250': '#960200'
}

fig_tir = px.bar(
    tir_df,
    x='cat_glucose',
    y='value',
    color='cat_glucose',
    facet_col='group',
    facet_col_wrap=2,
    text='percent_label',
    category_orders={'cat_glucose': bg_categories, 'group': ['Last Day', 'Last Week', 'Last Month', 'Last 3 Months']},
    color_discrete_map=bg_colors
)

fig_tir.update_traces(
    textposition='outside',
    cliponaxis=False,
    hovertemplate='Range: %{x}<br>Percentage: %{y:.0%}<extra></extra>'
)

fig_tir.update_yaxes(
    matches=None,
    tickformat='.0%',
    title=None
)

fig_tir.update_xaxes(
    title=None,
    tickangle=-45
)

fig_tir.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

fig_tir.update_layout(
    showlegend=False,
    bargap=0.3,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=10),
    margin=dict(t=40, r=20, b=40, l=40),
    height=400
)

# Recent CGM values chart (Interactive Plotly)
curr_df['date_local'] = curr_df['date'].dt.tz_convert('Europe/Madrid')

fig_recent = px.line(
    curr_df,
    x='date_local',
    y='sgv',
    color='device' if curr_df.device.nunique() > 1 else None,
    markers=True,
    hover_data={'date_local': '|%H:%M', 'sgv': True, 'device': True}
)

fig_recent.update_traces(
    mode='lines+markers',
    line=dict(width=2),
    marker=dict(size=6)
)

fig_recent.add_hline(
    y=TARGET_LOW,
    line_dash='dash',
    line_color='lightgreen',
    opacity=0.7,
    annotation_text='Low',
    annotation_position='right'
)

fig_recent.add_hline(
    y=TARGET_HIGH,
    line_dash='dash',
    line_color='lightgreen',
    opacity=0.7,
    annotation_text='High',
    annotation_position='right'
)

fig_recent.update_xaxes(title=None)
fig_recent.update_yaxes(title='Glucose [mg/dl]')

fig_recent.update_layout(
    legend_title_text='Device' if curr_df.device.nunique() > 1 else None,
    hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=10),
    margin=dict(t=40, r=20, b=40, l=60),
    height=400
)

# Site Definition

st.set_page_config('SugarBoard 📈', layout='wide')
col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
with col1:
    st.title('SugarBoard 📈')
    st.markdown('##### Bringing you closer to your own continous glucose monitoring data.')
with col2:
    st.header(f'Last value: **{last_value["sgv"]}** mg/dl {curr_dir}')

with col3:
    sel_initial_date = st.date_input('Initial date', value=datetime.datetime.now() - datetime.timedelta(days=90))
with col4:
    sel_final_date = st.date_input('Final date', value=datetime.datetime.now())

sel_url = f'https://{SITE}/api/v1/entries/sgv.json?&find[dateString][$gte]={sel_initial_date}' \
          f'&find[dateString][$lte]={sel_final_date}&count={MAX_VALUES}'

sel_df = (pd.DataFrame(requests.get(sel_url).json())
          .assign(date=lambda dd: pd.to_datetime(dd.dateString))
          [['date', 'sgv', 'device']]
          .set_index('date')
          .resample('5 min')
          ['sgv']
          .mean()
          .reset_index()
         )

sel_df_quantiles = (sel_df
                     .assign(hour=lambda dd: dd.date.dt.hour + dd.date.dt.minute / 60)
                     .groupby('hour')
                     .sgv
                     .agg([('median', np.median),
                           ('q90', lambda x: x.quantile(.9)),
                           ('q10', lambda x: x.quantile(.1)),
                           ('q25', lambda x: x.quantile(.25)),
                           ('q75', lambda x: x.quantile(.75))])
                     .rolling(10, center=True)
                     .mean()
                     .dropna()
                     .reset_index()
)

# Glucose patterns chart (Interactive Plotly)
fig_patterns = go.Figure()

# Add 10th-90th percentile ribbon (lightest)
fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q90'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q10'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 255, 255, 0.2)',
    line=dict(color='rgba(255, 255, 255, 0.2)'),
    name='10th-90th percentile',
    hoverinfo='skip'
))

# Add 25th-75th percentile ribbon (darker)
fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q75'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['q25'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 255, 255, 0.5)',
    line=dict(color='rgba(255, 255, 255, 0.5)'),
    name='25th-75th percentile',
    hoverinfo='skip'
))

# Add median line
fig_patterns.add_trace(go.Scatter(
    x=sel_df_quantiles['hour'],
    y=sel_df_quantiles['median'],
    mode='lines',
    line=dict(color='black', width=2),
    name='Median',
    hovertemplate='%{x:.1f}h: %{y:.0f} mg/dl<extra></extra>'
))

# Add target range lines
fig_patterns.add_hline(
    y=TARGET_LOW,
    line_dash='dash',
    line_color='lightgreen',
    opacity=0.7
)

fig_patterns.add_hline(
    y=TARGET_HIGH,
    line_dash='dash',
    line_color='lightgreen',
    opacity=0.7
)

fig_patterns.update_xaxes(
    title=None,
    tickvals=[3, 9, 15, 21],
    ticktext=['3h', '9h', '15h', '21h']
)

fig_patterns.update_yaxes(
    title='Glucose [mg/dl]',
    range=[40, None]
)

fig_patterns.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    font=dict(color='white', size=10),
    margin=dict(t=40, r=20, b=40, l=60),
    height=400
)

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
