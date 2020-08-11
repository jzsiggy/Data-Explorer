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


'''
    ## Class **Imbalance**
'''

def show_imbalance():
    y_vals_imbalance = data[target].values
    fig = px.bar(
                x=np.unique(y_vals_imbalance), 
                y=[ np.sum(y_vals_imbalance==class_) / len(y_vals_imbalance) for class_ in np.unique(y_vals_imbalance) ],
                labels={
                        "x": 'class',
                        "y": 'percentage of observations',
                    },
            )
    fig.update_layout(title_text='Class imbalance')
    st.write(fig)

if approach == 'Classification':
    show_imbalance()
else:
    st.write('This option is only available for classification problems.')


'''
    ## Feature **Occurencies**
'''

def show_feat_occurencies():
    features_1 = st.multiselect(
        "Choose features", list(X_vals.columns), [], key='occurencies_bar_plot'
    )
    if not features_1:
        st.error("Please select at least one feature.")    

    for feat in features_1:
        fig = px.bar(
            x=np.unique(data[feat]), 
            y=[ data.loc[data[feat] == val][target].count() for val in np.unique(data[feat]) ],
            labels={
                     "x": feat,
                     "y": 'num observations',
                 },
        )
        fig.update_layout(title_text='{} - Number of Observations'.format(feat))
        st.write(fig)

st.write('Be sure to select categorical features')

show_feat_occurencies()


'''
    ## Data **Distribuition**
'''


def show_classif_distributions():
    features_1 = st.multiselect(
        "Choose features", list(X_vals.columns), [], key='density_plot'
    )
    if not features_1:
        st.error("Please select at least one feature.")    

    for feat in features_1:
        if to_encode_target:
            group_labels = target_encoder.classes_
        else:
            group_labels = [activity for activity in np.unique(y_vals)]

        fig = ff.create_distplot(
            [ X_vals.iloc[y_vals == activity][feat] for activity in np.unique(y_vals)], 
            group_labels = group_labels,
            bin_size=0.5,
            show_rug=False,
        )
        fig.update_layout(title_text='{} Distribution Density'.format(feat))
        st.write(fig)

def show_reg_distributions():
    features_1 = st.multiselect(
        "Choose features", list(X_vals.columns), [], key='density_plot'
    )
    if not features_1:
        st.error("Please select at least one feature.")    

    for feat in features_1:
        fig = px.bar(
            x=np.unique(data[feat]), 
            y=[ data.loc[data[feat] == val][target].mean() for val in np.unique(data[feat]) ],
            labels={
                     "x": feat,
                     "y": target,
                 },
        )
        fig.update_layout(title_text='Mean {0} by {1}'.format(target, feat))
        st.write(fig)

if approach == 'Classification':
  show_classif_distributions()
else:
  show_reg_distributions()



'''
    ## Data **Trends** and **Seasonality**
'''

@st.cache(allow_output_mutation=True)
def get_trends(period):
    ts = go.Figure()
    ts.add_trace( 
        go.Scatter(x=date_ft, y=y_vals) 
    )
    ts.update_layout(
        title_text='Time Series',
        xaxis_title="Date",
        yaxis_title=target,
    )

    decomposition = seasonal_decompose(y_vals, period=int(period))
    trend = go.Figure()
    trend.add_trace(
        go.Scatter(x=date_ft, y=decomposition.trend)
    )
    trend.update_layout(
        title_text='Trend',
        xaxis_title="Date",
        yaxis_title=target,
    )

    seasonality = go.Scatter(x=date_ft, y=decomposition.seasonal, name='Seasonality')
    residual = go.Scatter(x=date_ft, y=decomposition.resid, name='Residual')

    fig = go.Figure()
    fig.add_trace(seasonality)
    fig.add_trace(residual)
    fig.update_layout(
        title_text='Seasonality & Residuality',
        xaxis_title="Date",
        yaxis_title=target,
    )
    
    return ts, trend, fig
    

if approach == 'Time Series Regression':
    period = st.text_input('What is the seasonality period of your data?', 365)
    ts, trend, seasonal = get_trends(period)
    st.write(ts)
    st.write(trend)
    st.write(seasonal)
else:
    st.write('This is only available for time series regression tasks.')


'''
    ## **Visualizing** Class Separation Based on 2 or 3 features
'''


px.defaults.color_continuous_scale = px.colors.sequential.Pinkyl

if approach == 'Classification':
  X_usage, y_usage = RandomUnderSampler(random_state=0).fit_resample(X_vals, y_vals)
else:
  X_usage, y_usage = X_vals, y_vals

def plot_classification(feats):
    if len(feats) == 2:
        fig = px.scatter(X_usage, x=feats[0], y=feats[1], color=y_usage)
        fig.update_traces(marker=dict(
                              size=8,
                              opacity=0.8,
                              line=dict(
                                  width=2,
                                  color='DarkSlateGrey'
                              )
                          ),
        )
    if len(feats) == 3:
        fig = px.scatter_3d(X_usage, x=feats[0], y=feats[1], z=feats[2], color=y_usage)
        fig.update_traces(marker=dict(
                              size=5,
                              line=dict(
                                  width=2,
                                  color='DarkSlateGrey'
                              )
                          ),
        )
    st.plotly_chart(fig)

features_2 = st.multiselect(
    "Choose 2 or 3 features", list(X_vals.columns), [], key='feat_plot'
)
if len(features_2) == 2:
    plot_classification(features_2)
elif len(features_2) == 3:
    plot_classification(features_2)
else:
    st.error("Please select between 2 and 3 features.")

if approach == 'Classification':
    st.write('**Number of data points has been resampled to show an equal number of active and inactive molecules')


'''
    ## **Visualizing** Class Separation Based on Many Features with **Dimensionality Reduction**
'''


@st.cache
def get_reduced(feats, dimensions, algo):
    data_for_red = X_usage[feats]
    if algo == 'PCA':
        reducer = PCA(random_state=0, n_components=dimensions)
    if algo == 'LDA':
        reducer = LinearDiscriminantAnalysis(n_components=dimensions)
    if algo == 'TSN-e':
        reducer = TSNE(random_state=0, n_components=dimensions)
    if algo == 'LDA':
        return reducer.fit_transform(data_for_red, y_usage)
    return reducer.fit_transform(data_for_red)

features_for_reduction = st.multiselect(
    "Choose features", list(X_vals.columns), [], key='pca_selector'
)

three_d = st.checkbox(label='3D')
def plot_reduced():
    algo = st.radio(
    "Which algorithm do you want to use for dimensionality reduction?",
    ('PCA', 'LDA', 'TSN-e')
)
    if three_d:
        reduced_features = get_reduced(features_for_reduction, 3, algo).transpose()
        fig = px.scatter_3d(x=reduced_features[0], y=reduced_features[1], z=reduced_features[2],
                    color=y_usage)
        fig.update_traces(marker=dict(
                              size=5,
                              line=dict(
                                  width=2,
                                  color='DarkSlateGrey'
                              )
                          ),
        )
    else:
        reduced_features = get_reduced(features_for_reduction, 2, algo).transpose()
        fig = px.scatter(x=reduced_features[0], y=reduced_features[1],
                    color=y_usage)
        fig.update_traces(marker=dict(
                              size=8,
                              opacity=0.8,
                              line=dict(
                                  width=2,
                                  color='DarkSlateGrey'
                              )
                          ),
        )

    st.plotly_chart(fig)

if len(features_for_reduction) < 3:
    st.error("Please select at least 3 features.")

elif approach == 'Time Series Regression':
    if ds in features_for_reduction:
        st.error("Date-type variable cannot be present in PCA transform.")
    else:
        plot_reduced()
else:
    plot_reduced()

if approach == 'Classification':
    st.write('**Number of data points has been resampled to show an equal number of active and inactive molecules')


'''
    ## Understanding **Feature Importances** Through Different Learning Algorithms
'''

if approach == 'Time Series Regression':
    X_usage = X_usage.drop(columns=[ds])
    st.write('The date column has been removed for this analysis.')


if approach == 'Classification':
    algorithm_table = {
        'Random Forest': RandomForestClassifier(),
        'Extremely Randomized Trees': ExtraTreesClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression()
    }
else:
    algorithm_table = {
        'Random Forest': RandomForestRegressor(),
        'Extremely Randomized Trees': ExtraTreesRegressor(),
    }

@st.cache
def get_importances(clf):
    model = clf.fit(X_usage, y_usage)
    try:
        importances = pd.Series(model.feature_importances_, index=X_usage.columns)
    except:
        importances = pd.Series(model.coef_[0], index=X_usage.columns)
    return importances
    
def show_ft_importances():
    model_for_selection = st.multiselect(
        "Choose algorithm", list(algorithm_table), ['Random Forest'], key='feat_importances'
    )
    if not model_for_selection:
        st.error("Please select at least one algorithm.")

    for estimator in model_for_selection:
        feat_importances = get_importances(algorithm_table[estimator])
        fig = plt.figure(figsize=(8,2.5))
        fig.add_axes(feat_importances.nlargest(20).plot(kind='bar'))
        fig.suptitle(estimator + ' Feature Importance', fontsize=16)

        st.write(fig)

show_ft_importances()

