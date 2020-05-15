# RESTful Machine Learning

## Introduction
This is a proof of concept application showing how machine learning models implemented as notebooks can be parameterized and executed using a RESTful service. By leveraging the power of MLflow for model managment, Papermill for dynamic execution and flexibility, and Fast API for the Python application, a generic architecture for model serving, online learning, and other microservices is possible. 


## MLflow and Papermill

Notebooks are parameterized using [Papermill](https://papermill.readthedocs.io/en/latest/index.html), and managed using [MLflow](https://mlflow.org/). MLflow projects are used as the entrypoint for the application, so that these modules can optionally be ran independently. This architecture allows for adding new modules and sevices in a simple way, while still maintaining the MLflow capabilities for model tracking and management.  

## Current Modules

### Sentiment Analysis Module

**Test Notebook**
Currently includes a test notebook to test Papermill execution logic. This will be used to run notebooks dynamically and call internal functions, for example, to train or request a prediction. 

**Sentiment Analysis in Keras**
A simple sentiment analysis notebook is included using the Keras library and IMDB dataset. Some of the accuracy and loss results are shown below:

![Accuracy](https://github.com/jcamstan3370/sentiment_analysis/blob/master/results/Accuracy.png)

![Loss](https://github.com/jcamstan3370/sentiment_analysis/blob/master/results/Loss.png)


