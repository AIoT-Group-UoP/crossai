from setuptools import setup

setup(
    name="crossai",
    version="0.1",
    packages=[
        "crossai",
        "crossai.ai",
        "crossai.ai.nn1d",
        "crossai.ai.nn2d"
    ],
    url="https://github.com/tzamalisp/crossai",
    license="Apache License 2.0",
    author="Pantelis Tzamalis, Panagiotis Kapetanidis, Fotios Kalioras, "
           "George Kontogiannis, Andreas Bardoutsos",
    author_email="tzamalis@ceid.upatras.gr",
    description="An open-source library that consists of functionalities "
                "capable of processing and developing ML pipelines for Time "
                "Series Analysis",
    install_requires=[
        "tensorflow==2.13.0",
        'tensorflow-metal==1.0.0; platform_system=="Darwin"',
        "tensorflow_addons>=0.21.0"
    ],
    python_requires=">=3.8"
)
