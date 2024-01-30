# CrossAI Library

This repository contains the open-source `crossai` library that incorporates functionalities capable of building 
end-to-end Artificial Intelligence pipelines for uni- or multi-axis Time-Series analysis. Moreover, it can handle 
processes that are related to Audio and Motion analysis. 

With the term "end-to-end AI pipeline", we mean actually, the development of a series of steps which has the 
following order:
1. data ingestion
2. data engineering
3. data preparation
4. AI model training
5. AI model evaluation on test set
6. AI model interpretation and evaluation of its robustness and trustworthiness on a pilot set (real-world scenario)

## Installation

To install the latest version of `crossai` library, run the following command:

`pip install crossai`

Note: The library supports **only Linux and Darwin (ARM64 MacOS with M1/M2 chips)** operating systems.

A useful guideline for installing TensorFlow in such a way that can exploit the GPU in 
M1/M2 chips can be found here:
* [How to Install TensorFlow GPU for Mac M1/M2 with Conda](https://www.youtube.com/watch?v=5DgWvU0p2bk)
* [Guideline](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-mac-metal-jan-2023.ipynb)
* [YAML conda environment](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/tensorflow-apple-metal.yml)

## AI Models

The library integrates various State-of-the-Art Neural Network architectures built on top of `TensorFlow`. These model
topologies can analyze time-series data in both the 1D and 2D domains. In particular, more details about the embedded 
to the library models can be found in the:
* `crossai/models/README.md`

## Data Processing

Below are the processing functionalities that the library contains.

### Processing Time Series

* Processing and labeling time-series

more info soon..

### Processing Audio

info soon..

### Processing Motion

info soon..

## Main Contributors

* [Pantelis Tzamalis](https://www.linkedin.com/in/pantelis-tzamalis/)
* [Andreas Bardoutsos](https://www.linkedin.com/in/andreasbardoutsos/)
* [Panagiotis Kapetanidis]()
* [Fotis Kalioras]()
* [George Kontogiannis]()

## Citing Cross-AI

If you use CrossAI library in your research or development, please use the following 
BibTeX entry:

```
@misc{
    CrossAI_Library, 
    author = {Tzamalis, Pantelis and Bardoutsos, Andreas and Kapetanidis, Panagiotis, and Kalioras, Fotis}, 
    url = {https://github.com/tzamalisp/crossai-lib}, 
    title = {{Cross-AI Library}}, 
    year = {2023}
}
```
