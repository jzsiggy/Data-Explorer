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
