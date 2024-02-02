from setuptools import setup, find_packages

setup(
    name='crossai',
    version='0.0.0.15',
    packages=['crossai', 'crossai.processing', 'crossai.preparation', 'crossai.processing.time_series',
              'crossai.models', 'crossai.models.nn1d', 'crossai.models.nn2d', 'crossai.perfomance',
              'crossai.perfomance.evaluation', 'crossai.processing.motion', 'crossai.processing.audio',
              'crossai.processing.signal', 'crossai.preparation.encoding', 'crossai.preparation.transformation',
              'crossai.preparation.scaler', 'crossai.preparation.augmentation'],
    url='https://github.com/tzamalisp/crossai-lib',
    license='GNU General Public License v3.0',
    author='Pantelis Tzamalis, Constantinos Tsakonas, Dimitrios Markantonatos, Andreas Bardoutsos',
    author_email='tzamalispantelis@gmail.com',
    description='A library of high-level functionalities capable of handling end-to-end Artificial Intelligence '
                'pipelines for Time-Series analysis',
    install_requires=[ 
        "librosa==0.9.2",
        "alibi",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "resampy>=0.4.0",
        "scikit-learn>=1.1.3",
        "scipy>=1.8.0",
        "setuptools>=61.2.0",
        "tensorflow==2.11.0; platform_system=='Linux'",
        "tensorflow-macos==2.11.0; platform_system=='Darwin'",
        "tensorflow-metal==0.7.0; platform_system=='Darwin'",
        "tensorflow_addons>=0.17.1",
        "toml>=0.10.2",
        "tqdm>=4.64.1",
        "albumentations>=1.3.0",
        "audiomentations>=0.28.0",
        "tsaug>=0.1.0",
        "seaborn>=0.11.2",
        "matplotlib>=3.4.3",
        "nlpaug==1.1.11"
    ],
    python_requires=">=3.8"
)
