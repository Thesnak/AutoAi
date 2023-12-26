import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model, load_model,evaluate_model,plot_model,predict_model
with st.sidebar:
    st.image("AutoAi.png")
    st.title('AutoAi')
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Evaluating", "Test", "Download"])
    st.info("AUTOMATION FOR EVER")
target=''
if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv')
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
if choice == 'Upload':
    st.title('Upload Your Dataset Here')
    file = st.file_uploader("Upload Your Dataset here")
    st.write("Note that you upload the dataset and AutoAi will split it for you")
    if file:
        df = pd.read_csv(file, index_col=None)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
        st.write("Train")
        st.dataframe(train_data)
        st.write('Test')
        st.dataframe(test_data)
        df.to_csv('data.csv', index=None)
        train_data.to_csv('train.csv', index=None)
        test_data.to_csv('test.csv', index=None)

if choice == "Profiling":
    st.title('Automated Exploratory Data Analysis')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Modelling":
    st.title("Machine Learning Classification ==============>")
    target = st.selectbox("Select Your Target", df.columns)
    setup(train_data, target=target, silent=True)
    setup_train = pull()
    st.info("This is The ML Classification Experiment settings")
    st.dataframe(setup_train)
    best_model = compare_models()
    compare_train = pull()
    st.info("This is The ML Classification Model")
    st.dataframe(compare_train)
    best_model
    plot_model(best_model, plot='auc' , save=True)
    plot_model(best_model, plot='pr' , save=True)
    plot_model(best_model, plot='confusion_matrix',save=True)
    save_model(best_model, 'best_model')
if choice == "Evaluating":
    if os.path.exists('AUC.png'):
        st.title("AUC :")
        st.image(image="AUC.png")
        st.title("Precision & Recall :")
        st.image(image="Precision Recall.png")
        st.title("Confusion Matrix :")
        st.image(image="Confusion Matrix.png")
if choice == "Test":
    if os.path.exists('best_model.pkl'):
        model = load_model('best_model')
        prediction = predict_model(model,data=test_data)
        st.dataframe(prediction)



    pass
if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model", f, 'trained_model.pkl')
