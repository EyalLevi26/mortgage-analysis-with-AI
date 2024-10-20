from setuptools import setup, find_packages

setup(
    name='mortgage_analysis',  # Replace with your project name
    version='0.1.0',
    author='Eyal Levi',  # Replace with your name
    author_email='eyal260290@gmail.com',  # Replace with your email
    description='AI-powered mortgage data analysis using time series forecasting \
        (ARIMA/SARIMAX), stationarity tests, and model optimization.\
        Includes data preprocessing and visualization for predictive insights.',
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        'pmdarima',  # List the main dependencies here
        'plotly',
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'statsmodels',
        'tqdm',
        'tabloo',
        'openpyxl',
        'argparse',
        'pyinstaller',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license if needed
    ],
)
