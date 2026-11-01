# Import Streamlit to create the web application interface
import streamlit as st

# Import pipeline to load and run Hugging Face models
from transformers import pipeline


# Configure the browser page title and layout
st.set_page_config(
    page_title="BERT vs GPT-2 Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# Display the main application heading and introduction
st.title("BERT vs GPT-2 Sentiment Analysis")
st.write("Compare two fine-tuned AI models on movie reviews.")


# Store the Hugging Face repository IDs of our trained models
MODELS = {
    "DistilBERT": "itsfareed120/distilbert-sentiment",
    "DistilGPT-2": "itsfareed120/distilgpt2-sentiment"
}


# Load the selected model and cache it to avoid repeated downloads
@st.cache_resource
def load_model(model_name):

    # Load the model and its corresponding tokenizer
    classifier = pipeline(
        "text-classification",
        model=MODELS[model_name],
        tokenizer=MODELS[model_name]
    )

    return classifier


# Create a text area where users enter their movie reviews
review = st.text_area(
    "Enter your movie review:",
    placeholder="Write your movie review here..."
)


# Allow users to select one model or compare both models
selected_model = st.selectbox(
    "Select a model:",
    ["DistilBERT", "DistilGPT-2", "Compare Both"]
)


# Create a button to start sentiment analysis
if st.button("Analyze Sentiment"):

    # Check whether the user entered any text
    if not review.strip():

        # Display a warning if the input is empty
        st.warning("Please enter a movie review.")

    else:

        # Select the models that need to perform prediction
        if selected_model == "Compare Both":
            models_to_run = ["DistilBERT", "DistilGPT-2"]
        else:
            models_to_run = [selected_model]

        # Run sentiment analysis using the selected models
        for model_name in models_to_run:

            # Display a loading message while the model is running
            with st.spinner(f"Loading {model_name}..."):

                # Load the selected Hugging Face model
                classifier = load_model(model_name)

                # Predict the sentiment of the entered review
                result = classifier(
                    review,
                    truncation=True,
                    max_length=256
                )[0]

            # Get the predicted classification label
            label = result["label"]

            # Convert the model's label into readable sentiment
            sentiment = (
                "Positive"
                if label in ["LABEL_1", "1"]
                else "Negative"
            )

            # Convert the prediction score into a percentage
            confidence = result["score"] * 100

            # Display the model name and prediction results
            st.subheader(model_name)
            st.write(f"**Prediction:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Display a progress bar representing the prediction score
            st.progress(result["score"])
