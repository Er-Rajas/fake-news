# Fake News Detection System

This project implements a fake news detection system using machine learning models with text vectorization.

## Overview

The fake news detection system utilizes machine learning techniques to classify news articles as either real or fake based on textual features. The system employs a vectorization approach coupled with a classification model to achieve accurate predictions.

## Directory Structure

```
fake_news_detection/
│
├── News_dataset/
│   ├── fake.csv
│   ├── true.csv
│
├── model/
│   ├── fake_news_model.pkl
│   ├── vectorizer.pkl
|
├── inference.py
├── train_model.py
├── preprocessing.py
|   
├── README.md
├── requirements.txt
├── app.py
```

## Setup

1. **Install Dependencies:**
   
   Install required Python packages using pip:

   ```sh
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   
   Execute the training script to train and save the classification model and vectorizer:

   ```sh
   python train_model.py
   ```

   This script preprocesses the data, performs vectorization, trains the model, and saves the trained model (`fake_news_model.pkl`) and vectorizer (`vectorizer.pkl`) in the `model/` directory.

3. **Run the Flask API:**
   
   Start the Flask API to serve predictions:

   ```sh
   python app.py
   ```

   The API runs locally on port 5000 by default. Adjust the port as needed in `app.py`.

## API Usage

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**

  ```json
  {
      "text": "Your news text here"
  }
  ```

- **Response:**

  ```json
  {
      "prediction": "News is true" or "News is fake"
  }
  ```

## Example Request

Using `curl`:

```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "NASA\'s Perseverance rover successfully lands on Mars. The rover will now begin its mission to search for signs of ancient life and collect samples for future return to Earth. This marks a significant achievement in space exploration and a milestone for NASA\'s Mars Exploration Program."}'
```

## Model and Vectorization

- **Model:** `fake_news_model.pkl`
  - Trained machine learning model for fake news classification.
  
- **Vectorizer:** `vectorizer.pkl`
  - Vectorizer used for text preprocessing and feature extraction.

## Dependencies

List of dependencies required to run the project:

- Flask
- Pandas
- NumPy
- Scikit-learn

## Model Evaluation

The model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

These metrics provide insights into the performance and effectiveness of the model in detecting fake news.

## Contribution

Feel free to fork the repository and submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Notes

- Ensure the datasets (`fake.csv`, `real.csv`) are placed in the `News_dataset/` directory before running `train_model.py`.
- Adjust parameters and model architecture in `train_model.py` for further experimentation and improvement.
- Handle errors and exceptions in `app.py` for robust API performance.

---

