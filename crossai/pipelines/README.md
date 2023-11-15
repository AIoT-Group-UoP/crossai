# Pipelines

## TimeSeries

This module contains the classes and functions to create a pipeline for
generic time series data:

* TimeSeries: CrossAI TimeSeries Class which contains the data and labels of the audio dataset. 
* ToPandas: Pipeline Class that can finalize a pipeline and return data as a pandas Dataframe. 
* Transformer: Creates a scikit-learn Transformer class for the pipeline. 
* PadOrTrim: Pipeline class to pad or trim the data in order for it to have the same dimensions. 
* labelEncoder: Pipeline class to encode the labels.

## Motion

This module containes the classes and functions to create a pipeline for motion data (a subset of Tabular data):

* PureAccExtractor: Extract the pure acceleration from every 3 axis feature.

## Tabular

This module contains the classes and functions to create a pipeline for tabular data:

* Tabular: CrossAI Tabular Class which contains the data, instances and labels of the multiaxial dataset
* MagnitudeExtractor: Extract the magnitude of from every feature using L2 norm.
* MultiAxisSlidingWindow: Pipeline Class to create a sliding window for multiaxial data.
* AxisToModelShape: Pipeline Class to reshape the multiaxial data to the shape of the model.
