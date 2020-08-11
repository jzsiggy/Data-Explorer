import streamlit as st
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE, SVMSMOTE, RandomOverSampler

from imblearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier,
                              RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
                             )
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from statsmodels.tsa.seasonal import seasonal_decompose

import random

'''
    # Interactive Data Exploration with Streamlit and Sklearn
    ##
'''

# st.sidebar.markdown('# Tools')
# st.sidebar.markdown('##')
# st.sidebar.checkbox('Class Imbalance', value=True)
# st.sidebar.checkbox('Data Distribution', value=True)
# st.sidebar.checkbox('Trends', value=True)
# st.sidebar.checkbox('Class Separation', value=True)
# st.sidebar.checkbox('PCA', value=True)
# st.sidebar.checkbox('Feature Importances', value=True)
# st.sidebar.checkbox('Model Builder', value=True)


@st.cache(suppress_st_warning=True)
def load_data(file):
    data = pd.read_csv(file)

    pct_null = data.isna().sum() / len(data)
    missing_features = pct_null[pct_null > 0.7].index
    data = data.drop(missing_features, axis=1)

    num_imputer = SimpleImputer()
    cat_imputer = SimpleImputer(strategy='constant')

    categorical_fts = data.select_dtypes(include='object')
    numerical_fts = data.drop(columns=categorical_fts.columns)

    if numerical_fts.columns.size:
        numerical_fts = pd.DataFrame(num_imputer.fit_transform(numerical_fts),columns = numerical_fts.columns)
    if categorical_fts.columns.size:
        categorical_fts = pd.DataFrame(cat_imputer.fit_transform(categorical_fts),columns = categorical_fts.columns)

    return categorical_fts.join(numerical_fts)


'''
    ## The **Data**:
'''


def display_dataframe_quickly(df, max_rows=50, **st_dataframe_kwargs):
    n_rows = len(df)
    if n_rows <= max_rows:
        st.write(df)
    else:
        start_row = st.slider('Start row', 0, n_rows - max_rows)
        end_row = start_row + max_rows
        df = df[start_row:end_row]

        if type(df) == np.ndarray:
            df = pd.DataFrame(df)
            df.index = range(start_row,end_row)

        st.dataframe(df, **st_dataframe_kwargs)
        st.text('Displaying rows %i to %i of %i.' % (start_row, end_row - 1, n_rows))

st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Choose a CSV file", type="csv")

if file is None:
    st.error('Please select a CSV file')
    st.stop()

raw_data = load_data(file)
data = raw_data.copy()
display_dataframe_quickly(data)

st.text('Columns with more than 80% NaN values have been removed')

approach = st.radio(
    "Is this a regression or classification task?",
    ('Regression', 'Classification', 'Time Series Regression')
)

if approach == 'Time Series Regression':
    ds = st.text_input("Date variable", 'ds')
    date_ft = pd.to_datetime(data[ds], infer_datetime_format=True)
    date_ft = date_ft.dt.tz_localize(tz='US/Eastern')
    data[ds] = date_ft

    diverse = st.checkbox('Does your data have more than one time series?')
    if diverse:
        column = st.multiselect(
            "Select a column to choose a time series from.", 
            list(data.columns), [], key='ts_analyze_col'
        )
        if len(column) != 1:
            st.error('You must choose one column for selecting your time series.')
            st.stop()
        
        col = column[0]
        if len(np.unique(data[col])) > 500:
            st.error('Too many distinct values to choose from')
            st.stop()

        value = st.multiselect(
            'Choose the value you will analyze',
            np.unique(data[col]), [], key='ts_analyze_val'
        )
        if len(value) != 1:
            st.error('You must choose exactly one value.')
            st.stop()

        data = data[ data[col].isin(value) ]


target = st.text_input("Target variable", 'target')

if approach == 'Classification':
    to_encode_target = st.checkbox('Do you wish to encode the target variable? (If the target variable is categorical, it must be encoded)')
    if to_encode_target:
        target_encoder = LabelEncoder()
        y_vals = target_encoder.fit_transform(data[target])
    else:
        y_vals = data[target].values
    X_vals = data.drop(columns=[target])

if approach != 'Classification':
    st.write('Trim your data')
    boundaries = st.slider(
        'Select target variable range (quantile)', 
        0, 
        100,
        ( 0 , 100 )
    )

    upper_limit = boundaries[1] / 100
    lower_limit = boundaries[0] / 100

    data = data[data[target] <= data[target].quantile(upper_limit)]
    data = data[data[target] >= data[target].quantile(lower_limit)]

    data = data.reset_index().drop(columns=['index'])

    y_vals = data[target].values
    X_vals = data.drop(columns=[target])

features_encode = st.multiselect(
    "Are there any categorical features you wish to encode? (Categorical features that are not encoded will be dropped.)", 
    list(X_vals.select_dtypes(include='object').columns), [], key='encode_fts'
)

categorical_fts = X_vals.select_dtypes(include='object')
numerical_fts = X_vals.drop(columns=categorical_fts.columns)
encoder = OrdinalEncoder()
categorical_fts_encoded = pd.DataFrame(encoder.fit_transform(categorical_fts[features_encode]), columns=features_encode)
X_vals = categorical_fts_encoded.join(numerical_fts)

features_drop = st.multiselect(
    "Are there any othr features you wish to drop", 
    list(X_vals.columns), [], key='drop_fts'
)

X_vals = X_vals.drop(columns=features_drop)

will_scale = st.checkbox('Do you want to scale your data?')
if will_scale:
    scaler = StandardScaler()
    X_vals = pd.DataFrame(scaler.fit_transform(X_vals),columns = X_vals.columns)

