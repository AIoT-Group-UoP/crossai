from setuptools import setup

setup(
    name="crossai",
    version="0.1.1.1",
    packages=[
        "crossai",
        "crossai.ai",
        "crossai.ai.nn1d",
        "crossai.ai.nn2d",
        "crossai.loader",
        "crossai.loader.pilot",
        "crossai.pipelines",
        "crossai.processing",
        "crossai.processing.audio",
        "crossai.processing.motion",
        "crossai.processing.tabular"
    ],
    url="https://github.com/tzamalisp/crossai",
    license="Apache License 2.0",
    author="Pantelis Tzamalis, Panagiotis Kapetanidis, Fotios Kalioras, "
           "George Kontogiannis, Andreas Bardoutsos",
    author_email="tzamalis@ceid.upatras.gr",
    description="An open-source library for processing and developing "
                "end-to-end ML pipelines for Time Series Analysis",
    install_requires=[
        "tensorflow==2.13.0",
        'tensorflow-metal==1.0.0; platform_system=="Darwin"',
        "tensorflow_addons>=0.21.0",
        "pandas==2.0.3",
        "scipy==1.11.1",
        "tqdm==4.65.0",
        "nlpaug==1.1.11",
        "librosa==0.9.2",
        "opencv-python==4.8.0.74",
        "scikit-learn==1.3.0",
        "seaborn>=0.12.2",
        "matchering==2.0.6"
    ],
    python_requires=">=3.10"
)
