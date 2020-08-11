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

