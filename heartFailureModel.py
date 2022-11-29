from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession \
    .builder \
    .appName('Heart_Spark') \
    .getOrCreate()

df = (spark.read
      .format("csv")
      .option('header', 'true')
      .load("datasets_heart.csv"))

dataset = df.select(col('age').cast('float'),
                    col('sex').cast('float'),
                    col('cp').cast('float'),
                    col('trestbps').cast('float'),
                    col('chol').cast('float'),
                    col('fbs').cast('float'),
                    col('restecg').cast('float'),
                    col('thalach').cast('float'),
                    col('exang').cast('float'),
                    col('oldpeak').cast('float'),
                    col('slope').cast('float'),
                    col('ca').cast('float'),
                    col('thal').cast('float'),
                    col('target').cast('float')
                    )

dataset = dataset.drop('thal')
dataset = dataset.drop('slope')
dataset = dataset.replace('?', None)\
    .dropna(how='any')

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
transformed_data = assembler.transform(dataset)

(training_data, test_data) = transformed_data.randomSplit([0.7, 0.3], seed=42)
rf = RandomForestClassifier(labelCol='target',
                            featuresCol='features',
                            maxDepth=5)

model = rf.fit(training_data)
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol='target',
    predictionCol='prediction',
    metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy)

model.save("model")

