
import streamlit as st
import pandas as pd
import scipy.io

import run

st.title('6G AI Trust Platform')

@st.cache
def load_data(csvFile):
    df = pd.read_csv(csvFile)
    return df

@st.cache
def process_error_metrics(result):
    # Create table for RMSE, MSE, and MAE
    df = pd.DataFrame()

    lst_model_names = result['models']
    lst_bfattack_rmse_list = result['bfattack_rmse_list']
    res_dct = {lst_model_names[i]: lst_bfattack_rmse_list[i] for i in range(0, len(lst_model_names))}
    row = pd.Series(res_dct,name='RMSE')
    df = df.append(row)
 
    lst_bfattack_mse_list = result['bfattack_mse_list']
    res_dct = {lst_model_names[i]: lst_bfattack_mse_list[i] for i in range(0, len(lst_model_names))}
    row = pd.Series(res_dct,name='MSE')
    df = df.append(row)
 
    lst_bfattack_mae_list = result['bfattack_mae_list']
    res_dct = {lst_model_names[i]: lst_bfattack_mae_list[i] for i in range(0, len(lst_model_names))}
    row = pd.Series(res_dct,name='MAE')
    df = df.append(row)

    return df

@st.cache
def process_prediction_results(result):
    # Plot actual and predictions
    df_predictions = pd.DataFrame();
    df_predictions['Actual'] = result['actual'].flatten().tolist()
    df_predictions['RNN_testPredict'] = result['RNN_testPredict'].flatten().tolist()
    df_predictions['LSTM_testPredict'] = result['LSTM_testPredict'].flatten().tolist()
    df_predictions['BiLSTM_testPredict'] = result['BiLSTM_testPredict'].flatten().tolist()
    df_predictions['GRU_testPredict'] = result['GRU_testPredict'].flatten().tolist()
    df_predictions['LSTM__Attention_testPredict'] = result['LSTM__Attention_testPredict'].flatten().tolist()
    df_predictions['BiLSTM__Attention_testPredict'] = result['BiLSTM__Attention_testPredict'].flatten().tolist()

    return df_predictions

@st.cache
def fetch_experiment_result(params):
    return run.run_experiment_1(params)

params = {}

uploaded_file_data = st.sidebar.file_uploader("Load Data")
if uploaded_file_data is not None:
    param_data = pd.read_csv(uploaded_file_data)
    st.write("Data loaded")
    st.write(param_data)
    params['param_data'] = param_data

uploaded_file_model = st.sidebar.file_uploader("Load Model")
if uploaded_file_model is not None:
    param_model = scipy.io.loadmat(uploaded_file_model)
    # st.write("Data loaded")
    params['param_model'] = param_model

# st.download_button('Download file', data)

param_attack_models = st.sidebar.multiselect('Attack Model',['Fast Gradient Sign Method (FGSM)', 
'Basic Iterative Method (BIM)', 
'Projected Gradient Descent (PGD)',
'Momentum Iterative Method (MIM)',
'Carlini & Wagner Attack (C&W)'])
params['param_attack_models'] = param_attack_models

param_attack_power = st.sidebar.selectbox('Attack Power', ['None', 'Low', 'Medium', 'High'])
params['param_attack_power'] = param_attack_power

param_defend_attack = st.sidebar.multiselect('Defend Attack', ['Adversarial Training' , 'Defensive Distillation'])
params['param_defend_attack'] = param_defend_attack


st.header('Experiment settings')
grid_search_selection = st.radio('Grid search', ['True', 'False'])
params['grid_search_selection'] = grid_search_selection


params['param_nr_epoch'] = st.slider('Number of epochs', 0, 150, 1)
params['param_batch_size'] = st.slider('Batch size', 8, 64, 32)

if st.button('Run Experiment'):
    st.write('Running experiment ...')
    result = fetch_experiment_result(params)
    st.write('Done')

    df_error_metrics = process_error_metrics(result)
    
    df_csv = df_error_metrics.to_csv().encode('utf-8')
    st.download_button(
    "Press to Download .csv format",
    df_csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )
    # show table
    st.write(df_error_metrics)

    df_predictions = process_prediction_results(result)
    st.line_chart(df_predictions)

    df_predictions_csv = df_predictions.to_csv().encode('utf-8')
    st.download_button(
    "Press to Download .csv format of predictions",
    df_predictions_csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

st.stop()
