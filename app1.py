# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:02:15 2023

@author: ichu9
"""

import streamlit as st
import torch
import requests
import pickle
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the saved model and tokenizer
saved_model_path = 'C:/Users/ichu9/model_and_tokenizer.pkl'  # Update with your saved model path
model_and_tokenizer = pickle.load(open(saved_model_path, 'rb'))
model = model_and_tokenizer['model']
tokenizer = model_and_tokenizer['tokenizer']

# Define class labels
class_labels = ["Human Generated", "Machine Generated"]

# Define the Streamlit app
def main():
    # Set the title and description of your app
    st.title("AI-Written or Human-Written")

    # Create a radio button for choosing input type
    input_type = st.radio("Choose input type:", ("Text", "URL"))

    if input_type == "Text":
        # Create a text input field for user input
        user_input = st.text_area("Enter text here:")

        # Create a button to trigger classification
        if st.button("Classify"):
            if user_input:
                # Tokenize the user input
                inputs = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")

                # Make predictions using the model
                with torch.no_grad():
                    outputs = model(**inputs)

                # Get the predicted class (assuming it's a classification task)
                predicted_class = torch.argmax(outputs.logits, dim=1).item()

                # Display the predicted class label
                predicted_label = class_labels[predicted_class]
                st.write(f"Predicted Class: {predicted_label}")

            else:
                st.warning("Please enter text for classification.")
    elif input_type == "URL":
        # Create an input field for the URL
        url_input = st.text_input("Enter URL here:")

        # Create a button to trigger classification
        if st.button("Classify"):
            if url_input:
                try:
                    # Fetch the content from the URL
                    response = requests.get(url_input)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    webpage_content = soup.get_text()

                    # Tokenize the webpage content
                    inputs = tokenizer(webpage_content, truncation=True, padding=True, return_tensors="pt")

                    # Make predictions using the model
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Get the predicted class (assuming it's a classification task)
                    predicted_class = torch.argmax(outputs.logits, dim=1).item()

                    # Display the predicted class label
                    predicted_label = class_labels[predicted_class]
                    st.write(f"Predicted Class: {predicted_label}")

                except Exception as e:
                    st.error(f"Error fetching and classifying content from the URL: {str(e)}")

            else:
                st.warning("Please enter a URL for classification.")

if __name__ == '__main__':
    main()


