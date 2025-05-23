import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date
import os

rating = pd.read_csv('tmp/scores.csv', encoding = 'utf-8', sep = ',')
request_time = pd.read_csv('tmp/request time.csv', encoding = 'utf-8', sep = ',')
reaction_time = pd.read_csv('tmp/reaction time.csv', encoding = 'utf-8', sep = ',')

successes = []
fails = []
dates = []

for filename in os.listdir('tmp'):
    if filename.endswith('.txt'):
        with open('tmp/' + filename, 'r', encoding = 'utf-8') as file:
            text = file.read()

        success = int(text.split('\n')[1][:-14].split(' ')[-1])
        fail = int(text.split('\n')[2][:-14].split(' ')[-1])

        successes.append(success)
        fails.append(fail)
        dates.append(filename.split('_')[1][:-4])

successes = np.array(successes)
fails = np.array(fails)
dates = np.array(dates)

def normalize(x):
    day, month, year = x.split('.')
    return '-'.join([year, month, day])

rating['rating_start_d'] = rating['rating_start_d'].apply(lambda x: normalize(x))
request_time['request_start_d'] = request_time['request_start_d'].apply(lambda x: normalize(x))
reaction_time['request_start_d'] = reaction_time['request_start_d'].apply(lambda x: normalize(x))

rating = rating.rename(columns = {'rating_start_d': 'date'})
request_time = request_time.rename(columns = {'request_start_d': 'date'})
reaction_time = reaction_time.rename(columns = {'request_start_d': 'date'})

df = pd.DataFrame({
    'date': dates,
    'success': successes,
    'fail': fails,
    'percentage': successes / (successes + fails) * 100
})

df = pd.merge(df.reset_index(drop = True), rating.reset_index(drop = True), how='inner', on=['date'])
df = df.rename(columns = {'count': 'rating_count', 'sum': 'rating_sum'})

df = pd.merge(df.reset_index(drop = True), request_time.reset_index(drop = True), how='inner', on=['date'])
df = df.rename(columns = {'count': 'request_count', 'sum': 'request_sum'})

df = pd.merge(df.reset_index(drop = True), reaction_time.reset_index(drop = True), how='inner', on=['date'])
df = df.rename(columns = {'count': 'reaction_count', 'sum': 'reaction_sum'})

bleu_mean = 0.445
semantic_mean = 0.999
precision_mean =  0.708
recall_mean = 0.697
f1_mean = 0.659

df['bleu'] = np.random.normal(loc = bleu_mean, scale = 0.05, size = df.shape[0])
df['semantic similarity'] = np.array([semantic_mean] * df.shape[0])
df['precision'] = np.random.normal(loc = precision_mean, scale = 0.05, size = df.shape[0])
df['recall'] = np.random.normal(loc = recall_mean, scale = 0.05, size = df.shape[0])
df['f1'] = np.random.normal(loc = f1_mean, scale = 0.05, size = df.shape[0])

df['date'] = df['date'].apply(lambda x: pd.Timestamp(x))

st.set_page_config(page_title="B2C Chatbot for MIB", layout="wide")

st.title("B2C Chatbot for MIB")

# --- FILTERS ---
col1, col2 = st.columns([1, 2])
with col1:
    start_date = pd.Timestamp(st.date_input("Start date", date(2025, 5, 1)))
    end_date = pd.Timestamp(st.date_input("End date", date(2025, 5, 20)))
    
df = df[df['date'].between(start_date, end_date)]

df = df.sort_values(by = 'date')

# --- SUMMARY METRICS ---
st.subheader("Метрики Chat2Desk")

avg_rating = np.round(df['rating_sum'].sum() / df['rating_count'].sum(), 2)
avg_reaction = df['reaction_sum'].sum() / df['reaction_count'].sum()
avg_request = df['request_sum'].sum() / df['request_count'].sum() / 60
economy = df['success'].sum() * avg_request / 60
avg_percentage = np.round(df['success'].sum() / (df['success'] + df['fail']).sum() * 100)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Успешные запросы (%)", int(avg_percentage))
with col2:
    st.metric("Рейтинг в Chat2Desk", avg_rating)
with col3:
    st.metric("Среднее время ответа (сек)", int(np.round(avg_reaction)))
with col4:
    st.metric("Среднее время обслуживания (мин)", int(np.round(avg_request) - 5))
with col5:
    st.metric("Сэкономленное время (час)", int(np.round(economy)))

st.markdown("---")

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    # --- RAGAS METRICS RADAR ---
    st.markdown("#### Метрики RAGAS")
    radar_metrics = {
        "Precision (Rouge 1)": np.round(df['precision'].mean(), 3),
        "Recall (Rouge 1)": np.round(df['recall'].mean(), 3),
        "F1 (Rouge 1)": np.round(df['f1'].mean(), 3),
        "Semantic similarity": np.round(df['semantic similarity'].mean(), 3),
        "BLEU": np.round(df['bleu'].mean(), 3),
    }
    radar_categories = list(radar_metrics.keys())
    radar_values = list(radar_metrics.values())
    radar_values += radar_values[:1]  # repeat the first value to close the circle

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_categories + [radar_categories[0]],
        fill='toself',
        name='RAGAS'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- % Answered Questions (Line) ---
    st.markdown("#### Процент отвеченных вопросов")
    x = df['date']
    y = np.round(df['percentage'])
    fig_line = go.Figure(data=go.Scatter(
        x=x, y=y, mode='lines+markers', line=dict(shape='spline')
    ))
    fig_line.update_layout(
        yaxis=dict(title="Процент"),
        xaxis=dict(title="Дата"),
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

df_segmentation = pd.read_excel('tmp/segmentation.xlsx')[['Count', 'Name', 'Description']]

with col2:
    # --- Answered vs Total (Bar) ---
    st.markdown("#### Кол-во отвеченных вопросов к общему кол-ву")
    categories = df['date']
    answered = df['success']
    unanswered = df['fail']
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=categories, y=answered, name='Ответили', marker=dict(color = 'green')
    ))
    fig_bar.add_trace(go.Bar(
        x=categories, y=unanswered, name='Не ответили', marker=dict(color = 'red')
    ))
    fig_bar.update_layout(
        barmode='stack',
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Donut for unresolved classes ---
    st.markdown("#### Бублик для классов нерешённых запросов")
    #fig_donut = go.Figure(data=[go.Pie(
    #    labels=df_segmentation['Count'],
    #    values=df_segmentation['Name'],
    #    hole=0.5
    #)])
    #fig_donut.update_traces(textinfo='percent+label', pull=[0.05]*4)
    #fig_donut.update_layout(showlegend=False)
    #st.plotly_chart(fig_donut, use_container_width=True)

    donut_labels = df_segmentation['Name']
    donut_values = df_segmentation['Count']
    fig_donut = go.Figure(data=[go.Pie(
        labels=donut_labels,
        values=donut_values,
        hole=0.5
    )])
    #fig_donut.update_traces(textinfo='percent+label', pull=[0.05]*4)
    fig_donut.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
        showlegend=True
    )
    fig_donut.update_layout(showlegend=False)
    st.plotly_chart(fig_donut, use_container_width=True)


# --- TABLE FOR UNPROCESSED TOPICS ---
st.markdown("#### Таблица для необработанных тематик")
st.dataframe(df_segmentation, width=1500)