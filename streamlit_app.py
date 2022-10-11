import datetime

import numpy as np
import pandas as pd
import plotnine as p9
import pytz
import requests
import streamlit as st

from src.utils import  mean_glucose_to_hba1c, inject_align_style
from mizani.formatters import date_format, percent_format

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

df = (pd.DataFrame(requests.get(url).json())
      .assign(date=lambda dd: pd.to_datetime(dd.dateString))
      [['date', 'sgv', 'device']]
      .set_index('date')
      .resample('5 min')
      ['sgv']
      .mean()
      .reset_index()
      .assign(cat_glucose=lambda dd: pd.cut(dd['sgv'], bins=[0, 50, 70, 150, 180, 250, np.inf],
                                            labels=['<50', '50-69', '70-150', '151-180', '181-250', '>250']))
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
            .apply(lambda dd: dd.cat_glucose.value_counts(True))
            .reset_index()
            .assign(percent_label=lambda dd: (dd.cat_glucose.round(2) * 100).astype(int).astype(str) + '%')
            .assign(group=lambda dd: pd.Categorical(dd.group, ordered=True,
                    categories=['Last Day', 'Last Week', 'Last Month', 'Last 3 Months']))
     )

f = (tir_df
     .pipe(lambda dd: p9.ggplot(dd)
                      + p9.aes(y='cat_glucose', fill='level_1')
                      + p9.geom_col(p9.aes(x='level_1'), width=.95)
                      + p9.geom_text(p9.aes(x='level_1', label='percent_label'), size=7, nudge_y=.05, color='white')
                      + p9.facet_wrap('group', ncol=2)
                      + p9.guides(fill=False)
                      + p9.labs(x='', y='', fill='')
                      + p9.scale_y_continuous(labels=percent_format(), limits=(0, dd.cat_glucose.max() + .1))
                      + p9.scale_fill_manual(['#960200', '#CE6C47', '#49D49D', '#FFD046', '#CE6C47', '#960200'])
                      + p9.theme_void()
                      + p9.theme(dpi=300,
                                 figure_size=(5, 3),
                                 text=p9.element_text(color='white'),
                                 axis_text_x=p9.element_text(rotation=60, va='center_baseline', size=8),
                                 axis_text_y=p9.element_blank(),
                                 strip_text=p9.element_text(size=11))
           )
     )

p = (curr_df
     .pipe(lambda dd:
           p9.ggplot(dd)
           + p9.aes('date', 'sgv')
           + p9.geom_point(size=.6, color='white')
           + p9.geom_line(p9.aes(group='device'), color='white')
           + p9.labs(x='', y='Glucose [mg/dl]')
           + p9.scale_x_datetime(labels=date_format('%H:%M', tz=pytz.timezone('CET')))
           + p9.annotate(geom='hline',
                         yintercept=[TARGET_LOW, TARGET_HIGH],
                         linetype='dashed',
                         color='lightgreen')
           + p9.theme_void()
           + p9.theme(dpi=300,
                      figure_size=(5, 3),
                      text=p9.element_text(color='white'),
                      title=p9.element_text(size=8),
                      axis_title_y=p9.element_text(rotation=90, size=9, va='bottom', margin={'r': 10}),
                      axis_text_y=p9.element_text(color='white', size=7, ha='right'),
                      axis_text_x=p9.element_text(size=7),
                      panel_grid=p9.element_text(color='white', alpha=.05))
           + p9.ylim(min(40, dd['sgv'].min()), max(200, dd['sgv'].max() + 10))
           )
     )

# Add device label in case data is being fetched simultaneously from more than one device
if curr_df.device.nunique() > 1:
    last_values_per_device = (curr_df
                              .groupby('device')
                              .apply(lambda dd: dd.sort_values('date').iloc[-1])
                              .reset_index(drop=True)
                              .assign(date=lambda dd: dd.date + pd.to_timedelta(('12 min')))
                              )
    p += p9.geom_text(p9.aes(label='device.str.split("-").str[0]'), data=last_values_per_device,
                      ha='center', size=6, color='white')

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

h = (sel_df_quantiles
     .pipe(lambda dd: p9.ggplot(dd)
       + p9.aes('hour', 'median')
       + p9.geom_ribbon(p9.aes(ymax='q90', ymin='q10'), alpha=.3, fill='white')
       + p9.geom_ribbon(p9.aes(ymax='q75', ymin='q25'), alpha=.7, fill='white')
       + p9.geom_line(color='black')
       + p9.annotate(geom='hline', yintercept=[TARGET_LOW, TARGET_HIGH], linetype='dashed', color='lightgreen')
       + p9.labs(y='Glucose [mg/dl]', x='')
       + p9.theme_void()
       + p9.theme(dpi=300,
                  figure_size=(5, 3),
                  text=p9.element_text(color='white'),
                  title=p9.element_text(size=8),
                  axis_title_y=p9.element_text(rotation=90, size=9, va='bottom', margin={'r': 10}),
                  axis_text_y=p9.element_text(color='white', size=7, ha='right'),
                  axis_text_x=p9.element_text(size=7),
                  panel_grid=p9.element_text(color='white', alpha=.05),
                  panel_border=p9.element_rect(fill='white')
                 )
       + p9.scale_y_continuous(limits=(40, None))
       + p9.scale_x_continuous(breaks=[3, 9, 15, 21], labels=[str(x) + 'h' for x in [3, 9, 15, 21]])
      )
)

st.markdown('---')

c1, c2, c3 = st.columns([3, 4, 4])
with c1:
    st.markdown(inject_align_style('#### Time in range (%) during last periods'), unsafe_allow_html=True)
    st.pyplot(p9.ggplot.draw(f))

with c2:
    st.markdown(inject_align_style('#### CGM values from the last 4 hours'), unsafe_allow_html=True)
    st.pyplot(p9.ggplot.draw(p))

with c3:
    st.markdown(inject_align_style('#### Glucose patterns from selected period'), unsafe_allow_html=True)
    st.pyplot(p9.ggplot.draw(h))
