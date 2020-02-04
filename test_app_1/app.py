# Based on:
# https://morioh.com/p/7066169a0314


# imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    # from https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
    # pretty print cm
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns_plot = sns.heatmap(cm, annot=annot, fmt='', ax=ax)

    return sns_plot.figure


st.title('Iris')
data = datasets.load_iris()
df = pd.DataFrame(data['data'],
        columns=['petal.length','petal.width','sepal.length','sepal.width'])
df['variety'] = data['target']
#df = pd.read_csv("iris.csv")

if st.checkbox('Show dataframe'):
    st.write(df)
    
st.subheader('Scatter plot')
species = st.multiselect('Show iris per variety?', df['variety'].unique())

col1 = st.selectbox('Which feature on x?', df.columns[0:4])
col2 = st.selectbox('Which feature on y?', df.columns[0:4])

new_df = df[(df['variety'].isin(species))]
st.write(new_df)

# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='variety')

# Plot!
st.plotly_chart(fig)
st.subheader('Histogram')
feature = st.selectbox('Which feature?', df.columns[0:4])
# Filter dataframe

new_df2 = df[(df['variety'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="variety", marginal="rug")
st.plotly_chart(fig2)
st.subheader('Machine Learning models')

features= df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
labels = df['variety'].values

# dataset splitting
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

# algorithms
alg = ['Decision Tree', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)

if classifier=='Decision Tree':

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    
    acc = dtc.score(X_test, y_test)
    
    st.write('Accuracy: ', acc)
    
    pred_dtc = dtc.predict(X_test)
    cm_dtc = cm_analysis(y_test,pred_dtc, dtc.classes_)
    #cm_dtc=confusion_matrix(y_test,pred_dtc)
    
    st.write('Confusion matrix: ', cm_dtc)

elif classifier == 'Support Vector Machine':

    svm=SVC()
    svm.fit(X_train, y_train)

    acc = svm.score(X_test, y_test)

    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)

    cm = cm_analysis(y_test,pred_svm, svm.classes_)
    #cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)