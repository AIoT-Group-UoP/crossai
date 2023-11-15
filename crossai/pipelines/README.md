# Pipelines

## TimeSeries

This module contains the classes and functions to create a pipeline for
generic time series data:

* TimeSeries: CrossAI TimeSeries Class which contains the data and labels of the audio dataset. 
* ToPandas: Pipeline Class that can finalize a pipeline and return data as a pandas Dataframe. 
* Transformer: Creates a scikit-learn Transformer class for the pipeline. 
* PadOrTrim: Pipeline class to pad or trim the data in order for it to have the same dimensions. 
* labelEncoder: Pipeline class to encode the labels.
* 