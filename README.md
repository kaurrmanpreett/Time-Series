# Time-Series

This project provides tools for time series analysis and forecasting using various machine learning models. It includes data preparation, model training, and evaluation modules.

## Project Structure

- `main.py`: Main script to run the project.
- `requirements.txt`: List of dependencies required for the project.
- `data/`: Directory containing data files for analysis.
  - `AAPL.csv`: Example dataset for analysis.
- `modules/`: Directory containing the project's modules.
  - `data_preparation.py`: Module for preparing the data for analysis.
  - `imports.py`: Module for importing necessary packages and libraries.
  - `model_evaluation.py`: Module for evaluating the trained models.
  - `model_training.py`: Module for training models on the prepared data.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/kaurrmanpreett/Time-Series.git

2. Create and activate a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
   pip install -r requirements.txt
   
5. To run the main script, use:
   python main.py

Modules
Data Preparation (data_preparation.py)
This module contains functions to clean and preprocess the data for analysis.

Imports (imports.py)
This module handles the import of necessary libraries and packages used across the project.

Model Evaluation (model_evaluation.py)
This module includes functions to evaluate the performance of the trained models.

Model Training (model_training.py)
This module provides functions to train various models on the prepared data.
   

