
import streamlit as st
import pandas as pd
import scipy.io

import run

st.title('6G AI Trust Platform')

@st.cache
def load_data(csvFile):
    df = pd.read_csv(csvFile)
    return df

# dfRaw = load_data()
# st.success('Read ' + str(dfRaw.shape[0]) + ' data points.')
# df = dfRaw


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


params['param_nr_epoch'] = st.slider('Number of epochs', 10, 150, 2)
params['param_batch_size'] = st.slider('Batch size', 8, 64, 32)
# params['param_start_idx'] = st.number_input('Pick a number', 0, 10000, 3000) #3000

if st.button('Run Experiment'):
    st.write('Run experiment with following data')
    result = run.run_experiment_1(params)
    st.write(result)
# else:
#     st.write('Goodbye')


st.stop()
agree = st.checkbox('I agree')
agree2 = st.checkbox('test')
if agree:
        st.write('Great!')

genre = st.radio(
    "What's your favorite movie genre",
    ('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")



age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')

values = st.slider('Select a range of values',
    0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)


txt = st.text_area('Text to analyze', 
    '''
     It was the best of times, it was the worst of times, it was
     the age of wisdom, it was the age of foolishness, it was
     the epoch of belief, it was the epoch of incredulity, it
     was the season of Light, it was the season of Darkness, it
     was the spring of hope, it was the winter of despair, ()
     ''')
st.write('Sentiment:', txt)

import datetime
d = st.date_input("When's your birthday",
    datetime.date(2019, 7, 6))
st.write('Your birthday is:', d)

t = st.time_input('Set an alarm for', datetime.time(8, 45))
st.write('Alarm is set for', t)


with st.echo():
    st.write('This code will be printed')

def get_user_name():
    return 'John'

with st.echo():
    # Everything inside this block will be both printed to the screen
    # and executed.

    def get_punctuation():
        return '!!!'

    greeting = "Hi there, "
    value = get_user_name()
    punctuation = get_punctuation()

    st.write(greeting, value, punctuation)

# And now we're back to _not_ printing to the screen
foo = 'bar'
st.write('Done!')

import time

my_bar = st.progress(0)

for percent_complete in range(100):
    my_bar.progress(percent_complete + 1)

with st.spinner('Wait for it...'):
    time.sleep(5)
    st.success('Done!')

# st.balloons()

st.error('This is an error')
st.warning('This is a warning')
st.info('This is a purely informational message')
st.success('This is a success message!')

my_placeholder = st.empty()

# Now replace the placeholder with some text:
my_placeholder.text("Hello world!")

# And replace the text with an image:
# my_placeholder.image(my_image_bytes)

import pandas as pd

import numpy as np

df1 = pd.DataFrame(np.random.randn(20, 10),
    columns=('col %d' % i for i in range(10)))

my_table = st.table(df1)

df2 = pd.DataFrame(np.random.randn(20, 10),
    columns=('col %d' % i for i in range(10)))

my_table.add_rows(df2)
# Now the table shown in the Streamlit app contains the data for
# df1 followed by the data for df2.

# Assuming df1 and df2 from the example above still exist...
my_chart = st.line_chart(df1)
my_chart.add_rows(df2)
# Now the chart shown in the Streamlit app contains the data for
# df1 followed by the data for df2.

# import matplotlib.pyplot as plt
# >>> import numpy as np
# >>>
# >>> arr = np.random.normal(1, 1, size=100)
# >>> plt.hist(arr, bins=20)
# >>>
# >>> st.pyplot()

# >>> import pandas as pd
# >>> import numpy as np
# >>> import altair as alt
# >>>
# >>> df = pd.DataFrame(
# ...     np.random.randn(200, 3),
# ...     columns=['a', 'b', 'c'])
# ...
# >>> c = alt.Chart(df).mark_circle().encode(
# ...     x='a', y='b', size='c', color='c')
# >>>
# >>> st.altair_chart(c, width=-1)
