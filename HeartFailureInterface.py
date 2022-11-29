from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
import streamlit as st
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession \
    .builder \
    .appName('Heart_Spark') \
    .getOrCreate()

model = RandomForestClassificationModel.load('./model')

st.write("""
# HEART FAILURE PREDICTION APP

This app predicts the likelihood of a person having an **Heart Attack** .

""")


st.sidebar.header('User Medical Records')

st.header('**Notice!**')
st.write("""
            1. age
            2. sex
            3. cp
            4. trestbps
            5. chol
            6. fbs
            7. restecg
            8. thalach
            9. exang
            10.oldpeak
            11.ca""")


def user_input_features():
    age = st.sidebar.slider('What is your Age?', 20, 100, 50)
    sex = st.sidebar.selectbox('What is your Sex?', ('Male', 'Female'))
    cp = st.sidebar.selectbox('What is the level of chest pain(CP) in your body?', (
        'typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    trestbps = st.sidebar.number_input('What is your blood pressure?', 0, 200)
    chol = st.sidebar.number_input(
        'What is your cholestoral in mg/dl?', 0, 240)
    fbs = st.sidebar.selectbox(
        'Do you have your fasting blood sugar superior at 120?', (True, False))
    restecg = st.sidebar.selectbox('What is the results of resting electrocardiographic?', (
        'normal', 'having ST-T wave abnormality(>0.05mV)', 'left ventricula hypertrophy'))
    thalach = st.sidebar.slider(
        'What is your maximun heart rate?', 30, 150, 50)
    exang = st.sidebar.selectbox(
        'Do you Have angina induced by exercise?', ('Yes', 'No'))
    oldpeak = st.sidebar.slider(
        'ST depression induced by excercise relative to rest?', 0.5, 7.0, 0.5)
    ca = st.sidebar.selectbox(
        'What is the number of major vessels?', ('0', '1', '2', '3'))
    data = {'age': age, 'sex': sex, 'cp': cp,  'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'ca': ca}
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()


def set_exang(row):
    if row["exang"] == 'yes':
        return 1
    else:
        return 0


input_df['exang'] = input_df.apply(set_exang, axis=1)


def set_ca(row):
    if row["ca"] == '0':
        return 0
    elif row["ca"] == '1':
        return 1
    elif row["ca"] == '2':
        return 2
    else:
        return 3


input_df['ca'] = input_df.apply(set_ca, axis=1)


def set_sex(row):
    if row["sex"] == 'Male':
        return 1
    else:
        return 0


input_df['sex'] = input_df.apply(set_sex, axis=1)


def set_fbs(row):
    if row["fbs"] == True:
        return 1
    else:
        return 0


input_df['fbs'] = input_df.apply(set_fbs, axis=1)


def set_cp(row):
    if row["cp"] == 'typical angina':
        return 1
    elif row["cp"] == 'atypical angina':
        return 2
    elif row["cp"] == 'non-anginal pain':
        return 3
    else:
        return 4


input_df['cp'] = input_df.apply(set_ca, axis=1)


def set_rest_ecg(row):
    if row["restecg"] == 'normal':
        return 0
    elif 'having ST-T wave abnormality(>0.05mV)':
        return 1
    else:
        return 2


input_df['restecg'] = input_df.apply(set_rest_ecg, axis=1)

dfs = spark.createDataFrame(input_df)
df = dfs.select(col('age').cast('float'),
                col('sex').cast('float'),
                col('cp').cast('float'),
                col('trestbps').cast('float'),
                col('chol').cast('float'),
                col('fbs').cast('float'),
                col('restecg').cast('float'),
                col('thalach').cast('float'),
                col('exang').cast('float'),
                col('oldpeak').cast('float'),
                col('ca').cast('float'),
                )
df = df.withColumn("target", lit(0))


required_features = ['age',
                     'sex',
                     'cp',
                     'trestbps',
                     'chol',
                     'fbs',
                     'restecg',
                     'thalach',
                     'exang',
                     'oldpeak',
                     'ca'
                     ]

assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(df)

predictions = model.transform(transformed_data)

st.dataframe(df)
st.dataframe(predictions)
