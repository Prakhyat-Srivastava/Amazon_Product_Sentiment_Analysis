from flask import Flask, request, jsonify, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved Logistic Regression model and TF-IDF vectorizer
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define a home route to render an HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form
    review_text = request.form['review']
    
    # Transform the input text using the saved TF-IDF vectorizer
    review_vector = vectorizer.transform([review_text])
    
    # Predict the sentiment probability using the saved model
    prediction_prob = model.predict_proba(review_vector)
    
    # Get the predicted class (1 = Positive, 0 = Negative)
    predicted_class = model.predict(review_vector)[0]
    
    # Calculate the percentage of surety
    surety_percentage = round(prediction_prob[0][predicted_class] * 100, 2)
    
    # Format confidence in small text
    confidence_text = f'<span class="confidence">(Confidence: {surety_percentage}%)</span>'
    
    # Determine sentiment and add emoji
    if predicted_class == 1:
        sentiment = f'Positive 😊 {confidence_text}'
    else:
        sentiment = f'Negative 😔 {confidence_text}'
    
    # Return the result, including the emoji, surety percentage, and retain the review text
    return render_template('index.html', prediction_text=sentiment, review_text=review_text)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
