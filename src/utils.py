from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


@st.cache_resource
def df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')


def viz_correlation_matrix(df: pd.DataFrame):

    corr = np.round(df.corr(numeric_only=True), 3)

    fig = px.imshow(
        corr,
        text_auto=True,
        height=720,
        width=1080,
    )

    return fig


def viz_chart(df: pd.DataFrame, x: str = None, color: str = None, nbins=1000, title: str = None,
               color_sequence: List[str] = None, legend_labels: List[str] = None,
               ascending: bool = False, pattern_shape: str = None):
    
    fig = px.histogram(
        df,
        x=x,
        color=color,
        nbins=nbins,
        pattern_shape=pattern_shape,
        title=title,
        color_discrete_sequence=color_sequence,
        template='seaborn',
        width=1080,
        height=720
    )

    fig.update_layout(bargap=0.2)
    fig.update_traces(marker_line_width=0.2,marker_line_color='black')

    if ascending:
        fig.update_layout(xaxis={'categoryorder':'total ascending'}) 

    if legend_labels:
        for idx, name in enumerate(legend_labels):
            fig.data[idx].name = name

    return fig


def viz_confusion_matrix(cm: np.array):

    x = ['false', 'true']
    y = ['predicted false', 'predicted true']
    fig = px.imshow(
        cm,
        x=x,
        y=y,
        text_auto=True,
        height=720,
        width=1080,
    )

    return fig


def viz_roc_auc(fprs: List[float], tprs: List[float], auc: float):

    fig = px.area(
    x=fprs, y=tprs,
    title=f'ROC Curve (AUC={auc:.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=1080, height=720
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(title_x=0.5)

    return fig


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=['TARGET'])

    # Имеет закрытые кредиты?
    X['HAS_CLOSED_LOANS'] = (X['LOAN_NUM_CLOSED'] >= 1).astype('int')

    # Разбиваем на группы по возрасту.
    X['AGE_LT_30'] = (X['AGE'] < 30).astype('int')
    X['AGE_BEETWEN_30_50'] = ((X['AGE'] >= 30) & (X['AGE'] < 50)).astype('int')
    X['AGE_GE_50'] = (X['AGE'] >= 50).astype('int')

    # Разбиваем на группы по персональному доходу
    X['PERSONAL_INCOME_BEETWEN_7.5_17.5'] = ((X['PERSONAL_INCOME'] >= 7500) & (X['PERSONAL_INCOME'] < 17500)).astype('int')
    X['PERSONAL_INCOME_BEETWEN_7.5_17.5'] = ((X['PERSONAL_INCOME'] >= 7500) & (X['PERSONAL_INCOME'] < 17500)).astype('int')
    X['PERSONAL_INCOME_BEETWEN_17.5_27.5'] = ((X['PERSONAL_INCOME'] >= 17500) & (X['PERSONAL_INCOME'] < 27500)).astype('int')
    X['PERSONAL_INCOME_GE_27.5'] = (X['PERSONAL_INCOME'] >= 27500).astype('int')

    # Больше 3 детей сожителей?
    X['CHILD_LE_3'] = (X['CHILD_TOTAL'] > 3).astype('int')
    X['DEPENDANTS_LE_3'] = (X['DEPENDANTS'] > 2).astype('int')

    # Имеет ли автомобиль?
    X['OWN_AUTO'] = (X['OWN_AUTO'] >=1).astype('int')

    # Формируем X и y
    X = X.drop(columns=['AGE',
                        'OWN_AUTO',
                        'GENDER',
                        'DEPENDANTS',
                        'CHILD_TOTAL',
                        'FL_PRESENCE_FL',
                        'REG_ADDRESS_PROVINCE',
                        'LOAN_NUM_TOTAL',
                        'LOAN_NUM_CLOSED',
                        'PERSONAL_INCOME'])
    y = df['TARGET']

    X = pd.get_dummies(X, columns=['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL'], drop_first=True, dtype='int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    t_enc = TargetEncoder(cols=['EDUCATION', 'FACT_ADDRESS_PROVINCE', 'MARITAL_STATUS', 'FAMILY_INCOME']).fit(X_train, y_train)

    X_train = t_enc.transform(X_train).reset_index(drop=True)
    X_test = t_enc.transform(X_test).reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def train_eval_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series):
    
    lr = LogisticRegression(class_weight='balanced', max_iter=100)
    lr.fit(X_train, y_train)

    test_pred = lr.predict(X_test)
    test_pred_proba = lr.predict_proba(X_test)

    tprs, fprs = [], []
    for thr in np.arange(0.1, 1.1, 0.1):
        thr_pred = (test_pred_proba[:, 1] > thr).astype('int64')
        tn, fp, fn, tp = confusion_matrix(y_test, thr_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tprs.append(tpr)
        fprs.append(fpr)

    f1 = f1_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)

    return f1, cm, auc, fprs, tprs