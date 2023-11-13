from setuptools import setup

setup(
    name="crossai",
    version="0.0.0.1",
    packages=[
        "crossai",
        "crossai.ai",
        "crossai.ai.nn1d",
        "crossai.ai.nn2d"
    ],
    install_requires=[
        "tensorflow==2.13.0",
        'tensorflow-metal==1.0.0; platform_system=="Darwin"',
        "tensorflow_addons>=0.21.0"
    ],
    python_requires=">=3.11.4"
)
