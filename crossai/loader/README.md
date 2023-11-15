# Data Loader

The Loader module facilitates the loading of external data into Dataframes, 
which can then be easily converted into CrossAI objects for processing 
pipelines. Currently, there are two main loader methods available:

* An audio loader that can read `.wav` files: `audio_loader`
* A tabular data loader that can read `.csv` files: `csv_loader`

## audio_loader

The `audio_loader` method is designed for processing audio files stored in a 
directory and organizing them into a dataframe row-wise. It offers flexibility 
in reading from either a single directory containing all the files or from 
multiple directories representing different classes.

### Directory Structure for Multiple Classes

When using the `audio_loader` for multiple classes, the directory structure 
should be as follows:

```plaintext
audio
  ├── class1
  │   ├── class1_1.wav
  │   ├── class1_2.wav
  │   ├── ...
  ├── class2
  │   ├── class2_1.wav
  │   ├── class2_2.wav
  │   ├── ...
  ├── ...
```

## csv_loader

The `csv_loader` method is designed for processing csv files stored in a 
directory and organizing them into a dataframe row-wise. It offers the same 
flexibility and functionality as the `audio_loader` method. The difference is 
that appends every column of the csvs in a dataframe row-wise, alongside an 
instance counter.

### Directory Structure for Multiple Classes

When using the `csv_loader` for multiple classes, the directory structure is 
the same as the `audio_loader` method.
