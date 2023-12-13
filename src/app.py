import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import utils

st.sidebar.title("Информация")
st.sidebar.info(
    """
    Этот дэшборд - моя попытка "потрогать", что такое streamlit. А заодно 
    попробовать обучить необучаемую(ого) модель логистической регрессии.
    """
)
st.sidebar.info(
    """
    - Автор: Витя Тихомиров
    - Мой гитхаб: https://github.com/onthebox
    """)

# Заглавие со смешной картинкой
st.title('Ну тыкни на рекламку...')
st.image('logo.jpg', caption='Кредит наличными', width=250)

# Описание работы
st.header('О чем речь?')
st.divider()

st.markdown(
    """
    У нас есть данные о многочисленных клиентах некоторого банка, в которых указано, помимо прочего,
    кликнул ли клиент на предложенную ему рекламу очень выгодного предложения от банка или нет. Наша задача
    изучить эти данные, а так же попытаться предсказать наиболее склонных к отклику клиентов.
    """
)

# Кнопка скачать таблицу
df = pd.read_csv('merged_data.csv')

csv = utils.df_to_csv(df)
st.download_button(
    label="Скачать данные в формате CSV (бесплатно)",
    data=csv,
    file_name='data.csv',
    mime='text/csv',
)

# Анализ данных
st.header('Взглянем на данные...')
st.divider()

st.markdown(f"Всего в данных {df.shape[0]} строк. Вот небольшой кусочек для примера:")

st.dataframe(df.sample(100), hide_index=True)

st.markdown(
    """
    Пропуски в данных у нас отсутствуют. Приступим к визуализации, попробуем составить представление 
    о взаимосвязи таргета с признаками. Для этого начнем с матрицы корреляции.
    """
)

st.plotly_chart(utils.viz_correlation_matrix(df), use_container_width=True)

st.markdown(
    """
    Чтож, в целом ничего экстраординарного. Однако видим (увеличьте, там правда видно), что 
    целевая переменная слабо коррелирует с остальными признаками. Наибольшая взаимосвязь
    наблюдается с признаком AGE, и PERSONAL_INCOME. Попробуем построить немного графиков,
    может быть найдем что-то интересное. 
    """
)

curr_fig = utils.viz_chart(
    df,
    x='AGE',
    color='GENDER',
    nbins=80,
    title='Распределение клиентов по возрасту и полу',
    color_sequence=['dodgerblue', 'pink'],
    legend_labels=['1, Male; target: 0', '1, Male; target: 1', '0, Female; target: 0', '0, Female; target: 1'],
    pattern_shape='TARGET'
)
st.plotly_chart(curr_fig, use_container_width=True)

st.markdown(
    """
    Видим, что клиентов мужчин несколько больше, чем женщин, что среди мужчин примерно равное количество клиентов всех 
    возрастов от 23 до 58 лет, количество мужчин младше или старше идет на спад. Женщин же больше всего в возрасте от 25 до 30 лет. 
    Однако целевая переменная, на вид, распределена примерно одинаково среди основной массы пользователей банка.
    """
)

curr_fig = utils.viz_chart(
    df,
    x='AGE',
    color='TARGET',
    nbins=10,
    title='Распределение клинетов по возрасту',
    color_sequence=['dodgerblue', 'green'],
    legend_labels=None,
    pattern_shape=None
)
st.plotly_chart(curr_fig, use_container_width=True)

st.markdown(
    """
    Больше всех на объявления откликаются клиенты в возрасте от 25 до 55 лет.
    """
)

curr_fig = utils.viz_chart(
    df,
    x='PERSONAL_INCOME',
    color='GENDER',
    nbins=120,
    title='Распределение клиентов по возрасту и доходам',
    color_sequence=['dodgerblue', 'pink'],
    legend_labels=['1, Male; target: 0', '1, Male; target: 1', '0, Female; target: 0', '0, Female; target: 1'],
    pattern_shape='TARGET'
)
st.plotly_chart(curr_fig, use_container_width=True)

st.markdown(
    """
    Основную массу клиентов составляют лица с доходами от 2.5к до 27к рублей (?), при этом абсолютное большинство из них 
    имеют доход от 7.5к до 12.5к. Среди этих людей, ожидаем, находится больше всего откликнувшихся на предложение.
    """
)

curr_fig = utils.viz_chart(
    df,
    x='FACT_ADDRESS_PROVINCE',
    color=None,
    nbins=100,
    title='Распределение клиентов по регионам',
    color_sequence=['dodgerblue', 'pink'],
    legend_labels=None,
    pattern_shape='TARGET',
    ascending=True
)
st.plotly_chart(curr_fig, use_container_width=True)

st.markdown(
    """
    Очень много клиентов нашего банка, значительно больше, чем в любом другом регионе, проживет в 
    Кемероской области и Алтайском крае. Что касается целевой переменной, то можно сказать, что для 
    каждого региона она соразмерна суммарному количеству жителей.
    """
)

curr_fig = utils.viz_chart(
    df,
    x='LOAN_NUM_TOTAL',
    color='TARGET',
    nbins=20,
    title='Распределение клинетов по количеству кредитов',
    color_sequence=['gray', 'green'],
    legend_labels=None,
    pattern_shape=None
)
st.plotly_chart(curr_fig, use_container_width=True)

curr_fig = utils.viz_chart(
    df,
    x='LOAN_NUM_CLOSED',
    color='TARGET',
    nbins=20,
    title='Распределение клинетов по количеству закрытых кредитов',
    color_sequence=['darkgray', 'green'],
    legend_labels=None,
    pattern_shape=None
)
st.plotly_chart(curr_fig, use_container_width=True)

st.markdown(
    """
    Основная масса клиентов, которые откликнулись (да и вообще основная масса) имеют не более одного кредита всего или 
    не более одного закрытого кредита.
    """
)

curr_fig = utils.viz_chart(
    df,
    x='CHILD_TOTAL',
    color='TARGET',
    nbins=15,
    title='Распределение клинетов по количеству детей',
    color_sequence=['dodgerblue', 'green'],
    legend_labels=None,
    pattern_shape=None
)
st.plotly_chart(curr_fig, use_container_width=True)

curr_fig = utils.viz_chart(
    df,
    x='DEPENDANTS',
    color='TARGET',
    nbins=10,
    title='Распределение клинетов по количеству иждивенцев',
    color_sequence=['dodgerblue', 'green'],
    legend_labels=None,
    pattern_shape=None
)
st.plotly_chart(curr_fig, use_container_width=True)

st.markdown(
    """
    Наибольшая масса клиентов имеют от 0 до 2 детей/иждивенцев, среди них же и наибольшее количество откликнувшихся.
    """
)

# Обучаем модель
st.header('Попробуем обучить модель')
st.divider()

pos, neg = df['TARGET'].sum(), (df['TARGET'] == 0).sum()

st.markdown(
    """
    Для предсказания будем использовать логистическую регрессию. Обратим внимания, что классы несбалансированы, положительных классов 
    всего {pos}, а отрицательных - {neg}.

    Предварительно мы добавим некоторые новые признаки в данные исходя из визуализаций выше, а именно:
    разобьем возраст и персональный доход клиента на интервалы в соответствии с ранее построенными графиками, 
    изменим столбец о налиции автомобиля так, чтобы он показывал только факт наличия (а не количество автомобилей), 
    а так же добавим столбец-индикатор, который показывает, есть ли у клиента закрытые кредиты. А еще некоторые столбцы удалим 
    за неинформативностью. И закодируем некоторые переменные Target'ом, а некоторые One-Hot'ом.

    На тест и трейн будем разбивать в пропорции 1 к 4.
    """.format(pos=pos, neg=neg)
)

X_train, X_test, y_train, y_test = utils.prepare_data(df)

st.markdown("""Вот так выглядят данные после обработки, готовы запихиваться в модель.""")
st.dataframe(X_train.sample(100), hide_index=True)

f1, cm, auc, fprs, tprs = utils.train_eval_model(X_train, X_test, y_train, y_test)

st.markdown(
    """
    f1 метрика по результатам нашей модели на тестовой выборке получилась {f1}. Не богато, но хотя бы
    что-то. Покажем результат на матрице ошибок:
    """.format(f1=f1)
)

st.plotly_chart(utils.viz_confusion_matrix(cm), use_container_width=True)

st.markdown(
    """
    В завершении построим ROC-кривую и посчитаем площадь под ней.
    """
)

st.plotly_chart(utils.viz_roc_auc(fprs, tprs, auc), use_container_width=True)

st.markdown(
    """
    Из-за большого дисбаланса классов и отсутствия сильных корреляций признаков с целевой переменной
    результаты мягко говоря не очень :( В целом, это пространство для размышлений - как сделать эту модель хоть сколько-нибудь
    лучше. А пока что у нас получилось значение AUC {auc:.4f}. Можно сказать, что это немного лучше 
    рандомного предикта, и это уже неплохо :)
    """.format(auc=auc)
)
