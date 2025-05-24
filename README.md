# Ecommerce Item Category Classification using TF-IDF

This project demonstrates how to classify ecommerce item descriptions into categories using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization method and various machine learning models.

## Dataset

The dataset used in this project is the [Ecommerce Text Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification) dataset from Kaggle. It contains descriptions of ecommerce items and their corresponding categories.

## Project Structure

The project is implemented in a Google Colab notebook and covers the following steps:

1.  **Data Loading and Preparation:**
    *   Downloading the dataset from Kaggle using the Kaggle API.
    *   Loading the data into a pandas DataFrame.
    *   Handling data cleaning and initial exploration (checking for missing values, data types, etc.).
    *   Adding appropriate column headers to the DataFrame.

2.  **Data Balancing:**
    *   Addressing class imbalance by upsampling minority classes to match the majority class count.

3.  **Text Preprocessing:**
    *   Defining a utility function using `spaCy` for text preprocessing.
    *   Steps include removing stop words, punctuation, and lemmatization.
    *   Applying the preprocessing function to the text data.

4.  **Model Training and Evaluation:**
    *   Splitting the data into training and testing sets.
    *   Implementing and evaluating different classification models:
        *   K-Nearest Neighbors (KNN)
        *   Random Forest
        *   Multinomial Naive Bayes
    *   Using `TfidfVectorizer` to convert text data into numerical features for the models.
    *   Evaluating the models using a classification report (precision, recall, f1-score).

## Technologies Used

*   Python
*   pandas
*   scikit-learn
*   spaCy
*   Kaggle API

## How to Run the Code

1.  Clone the repository to your local machine.
2.  Open the Google Colab notebook (`.ipynb` file).
3.  Obtain your Kaggle API token from your Kaggle account settings.
4.  Upload your `kaggle.json` file to the Colab environment when prompted.
5.  Run the cells in the notebook sequentially.

## Results

The classification report for each model is printed to the console after training and evaluation. This allows for comparison of the performance of the different models on this dataset.

## Future Improvements

*   Experiment with other text vectorization techniques (e.g., Word Embeddings like Word2Vec, GloVe).
*   Explore more advanced machine learning models or deep learning models (e.g., Recurrent Neural Networks, Transformers).
*   Perform hyperparameter tuning to optimize the performance of the chosen models.
*   Implement a more robust preprocessing pipeline.
