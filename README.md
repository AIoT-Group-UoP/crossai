# CrossAI

This repository contains the open-source `crossai` library, which includes functionalities capable of developing 
end-to-end Artificial Intelligence (AI) pipelines for uni- or multi-axis Time Series analysis. The library specifically 
incorporates components in the likes of:

1. Signal engineering: signal transformation, feature engineering, data augmentation
2. State-of-the-Art Neural Network models, capable of training with time series data
3. Explainable AI (XAI) modeling for evaluating the AI time series pilot data
4. Support of scikit-learn pipelines for the data processing, as well as the pilot evaluation components

The components have been built on top of `TensorFlow`, `scikit-learn`, `pandas`, `NumPy`, and `SciPy`, while the library focuses 
on state-of-the-art techniques for Time Series tasks, in both the data engineering and Neural Network models.

> With the term "end-to-end AI pipelines", we mean actually, the development of a series of steps which has the 
> following order: a) data ingestion (loading), b) data engineering, c) data preparation, d) AI model training 
> (and fine-tuning), and, e) AI model interpretation and evaluation of its robustness and trustworthiness on pilot data, 
> meaning data that emulates real-world scenarios.

## Components

The CrossAI library comprises the following components:

### Time Series Processing
> It provides processing tasks and functions that can be utilized both within and outside of pipelines for 
> signal transformations and feature engineering. It also includes domain-based functionalities for audio, motion, etc.

### Machine Learning
> It offers a collection of State-of-the-Art Neural Network architectures, built on top of `TensorFlow`. These model 
> topologies are designed to analyze Time Series data within both the 1D and 2D domains.

### Explainable AI (XAI)
> It delivers Explainable AI (XAI) mechanisms for interpreting the AI models' exports to actual exports of interest, 
> as well as functionalities and algorithms for measuring the models' Trustworthiness and Robustness.
> The implementation is based on the following study:
> Tzamalis, Pantelis, Andreas Bardoutsos, Dimitris Markantonatos, Christoforos Raptopoulos, Sotiris Nikoletseas, 
> Xenophon Aggelides, and Nikos Papadopoulos. "End-to-end Gesture Recognition Framework for the Identification of 
> Allergic Rhinitis Symptoms." In 2022 18th International Conference on Distributed Computing in Sensor 
> Systems (DCOSS), pp. 25-34. IEEE, 2022.

Next steps: Add Generative AI State-of-the-Art Neural Network Topologies.

## Citing CrossAI

If you use CrossAI in a scientific publication, we would appreciate citations, using the following Bibtex entry:

```text
@misc{,
    author = "{Tzamalis, Pantelis and Kapetanidis Panagiotis, and Kalioras, Fotios and Kontogiannis, George and 
    Bardoutos, Andreas}",
    title = "CrossAI: The Open Source library for Time Series Analysis",
    year = "2023",
    url = "https://github.com/tzamalisp/crossai",
    note = "[Online; accessed 14-November-2023]"
  }
```

## Team

#### Authors

The following individuals are currently core contributors to the development and maintenance of the library:

* [Pantelis Tzamalis](https://www.linkedin.com/in/pantelis-tzamalis/)
* [Panagiotis Kapetanidis](https://www.linkedin.com/in/kapetanidispanagiotis)
* [Fotios Kalioras](https://www.linkedin.com/in/fotis-kalioras)
* [George Kontogiannis](https://www.linkedin.com/in/georgios-kontogiannis)

#### Emeritus Contributor Experience Team:

The following people have been active in the contribution of previous versions of the library but no longer are engaging
in its development:

* [Andreas Bardoutsos](https://www.linkedin.com/in/andreasbardoutsos/)

The main design and idea have been created by Pantelis Tzamalis, as well as Andreas Bardoutsos in the early versions of
the library, and occurred from needs during the development of AI projects with regard to Time Series tasks 
(classification, regression, clustering, etc.).
