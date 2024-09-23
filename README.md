# Titanic Survival Prediction Using PySpark

This project demonstrates a binary classification approach using PySpark MLlib to predict the survival of passengers on the Titanic dataset. Various machine learning algorithms are implemented, including Logistic Regression, Decision Trees, Random Forest, Gradient-Boosted Trees, Support Vector Machine, and Naive Bayes.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build a binary classifier using PySpark to predict the survival of passengers on the Titanic. We explore various classification algorithms, evaluate their performance, and select the best model based on evaluation metrics.

## Setup and Installation
To run this project, you need to set up PySpark and Java. Below are the steps to set up the environment in Colab:

```bash
# Download Java and Spark
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
!tar xf spark-3.2.1-bin-hadoop3.2.tgz
!pip install -q findspark

# Set up the paths
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"

# Create a Spark session
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) # Property used to format output tables better
spark.conf.set("spark.sql.caseSensitive", True) # Avoid error "Found duplicate column(s) in the data schema"
spark

```

## Data Preparation
The dataset used in this project is the Titanic passenger data. After loading the data, several preprocessing steps are performed:

- Dropping columns with high missing values or low relevance.
- Imputing missing values for numerical and categorical variables.
- Encoding categorical variables using one-hot encoding.

## Feature Engineering
Feature engineering involves creating new features from existing data to improve the predictive power of the models:

- Creating dummy variables for categorical features like Embarked and Sex.
- Combining SibSp and Parch to create a FamilySize feature.

## Modeling
Several machine learning models are built using PySpark:

- **Logistic Regression**: A statistical model that predicts binary outcomes.
- **Decision Trees**: A tree-based model that splits data based on feature values.
- **Random Forest**: An ensemble method that combines multiple decision trees.
- **Gradient-Boosted Trees**: An ensemble method that builds trees sequentially to improve errors.
- **Support Vector Machine (SVM)**: Finds the optimal hyperplane to separate classes.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.

## Evaluation
Models are evaluated using several metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Area Under ROC Curve (AUC)

## Results
The performance of the models is summarized below:

| Model                     | AUC      | Accuracy | Precision | Recall   | F1       |
|---------------------------|----------|----------|-----------|----------|----------|
| Random Forest             | 0.892243 | 0.841530 | 0.841023  | 0.841530 | 0.841250 |
| Logistic Regression       | 0.869817 | 0.803279 | 0.808585  | 0.803279 | 0.805052 |
| Gradient-Boosted Trees    | 0.866362 | 0.803279 | 0.806547  | 0.803279 | 0.804519 |
| Support Vector Machine    | 0.854563 | 0.765027 | 0.767721  | 0.765027 | 0.766165 |
| Decision Tree             | 0.597458 | 0.836066 | 0.837356  | 0.836066 | 0.836607 |
| Naive Bayes               | 0.528357 | 0.699454 | 0.689202  | 0.699454 | 0.690602 |

Based on the above metrix, it is clearly eveident that Random forest outperformed others.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

