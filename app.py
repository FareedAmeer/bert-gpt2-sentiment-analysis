
# Import Streamlit to build the web application
import streamlit as st

# Import Hugging Face pipeline to run sentiment analysis
from transformers import pipeline


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sentiment AI | Model Arena",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# 2. CUSTOM CSS - MODERN DARK UI
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background:
        radial-gradient(ellipse at 10% 5%, #292015 0%, transparent 35%),
        radial-gradient(ellipse at 90% 90%, #17251c 0%, transparent 35%),
        #101210;
    color: #F5F5F0;
}

/* Main content spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Main heading */
.hero-title {
    font-size: 48px;
    font-weight: 850;
    letter-spacing: -2px;
    color: #F5F5F0;
    margin-bottom: 5px;
}

/* Highlighted heading text */
.hero-highlight {
    color: #C5F36A;
}

/* Subtitle */
.hero-subtitle {
    color: #A6AAA2;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Top label */
.eyebrow {
    color: #FF9D52;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

/* Information card */
.info-card {
    background: #1B1F1B;
    border: 1px solid #333A31;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}

.info-number {
    color: #C5F36A;
    font-size: 25px;
    font-weight: 800;
}

.info-label {
    color: #A6AAA2;
    font-size: 12px;
}

/* Section headings */
.section-title {
    color: #F5F5F0;
    font-size: 22px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 8px;
}

/* Text area */
.stTextArea textarea {
    background-color: #1B1F1B !important;
    color: #F5F5F0 !important;
    border: 1px solid #424A3F !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}

/* Select box */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1B1F1B;
    border-color: #424A3F;
    border-radius: 10px;
    color: white;
}

/* Main button */
.stButton > button {
    background: linear-gradient(90deg, #C5F36A, #A6D94F);
    color: #151A12;
    border: none;
    border-radius: 10px;
    padding: 13px 25px;
    font-weight: 800;
    font-size: 16px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #E0FF9B;
    color: #101210;
    transform: translateY(-2px);
}

/* Prediction result card */
.result-card {
    background: linear-gradient(145deg, #202620, #191D19);
    border: 1px solid #394437;
    border-radius: 18px;
    padding: 24px;
    margin-top: 12px;
    min-height: 190px;
}

/* Model name */
.model-name {
    color: #C5F36A;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Prediction label */
.prediction {
    font-size: 32px;
    font-weight: 850;
    margin-top: 15px;
    margin-bottom: 5px;
}

/* Confidence text */
.confidence {
    color: #B5B9B0;
    font-size: 14px;
}

/* Confidence progress track */
.progress-track {
    height: 9px;
    background: #343A32;
    border-radius: 20px;
    margin-top: 18px;
    overflow: hidden;
}

/* Confidence progress fill */
.progress-fill {
    height: 100%;
    border-radius: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #777E74;
    font-size: 12px;
    padding-top: 35px;
}

/* Horizontal divider */
hr {
    border-color: #333A31 !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.markdown("""
<div class="eyebrow">AI POWERED TEXT INTELLIGENCE</div>

<div class="hero-title">
    Sentiment <span class="hero-highlight">AI.</span>
</div>

<div class="hero-subtitle">
    Explore how two Transformer architectures understand movie reviews.
    Compare predictions with fine-tuned language models.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. PROJECT INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">02</div>
        <div class="info-label">Fine-tuned AI Models</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">IMDb</div>
        <div class="info-label">Movie Review Dataset</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">NLP</div>
        <div class="info-label">Sentiment Classification</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 5. HUGGING FACE MODEL REPOSITORIES
# --------------------------------------------------

MODELS = {
    "DistilBERT": "itsfareed120/distilbert-sentiment",
    "DistilGPT-2": "itsfareed120/distilgpt2-sentiment"
}


# --------------------------------------------------
# 6. LOAD AND CACHE THE SELECTED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_name):

    # Load the fine-tuned model and its tokenizer
    classifier = pipeline(
        "text-classification",
        model=MODELS[model_name],
        tokenizer=MODELS[model_name]
    )

    return classifier


# --------------------------------------------------
# 7. REVIEW INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">01 / Enter Your Movie Review</div>',
    unsafe_allow_html=True
)

review = st.text_area(
    "Your review",
    placeholder="Write or paste a movie review here... What did you think about the movie?",
    height=170,
    label_visibility="collapsed"
)


# --------------------------------------------------
# 8. MODEL SELECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">02 / Choose Your AI Model</div>',
    unsafe_allow_html=True
)

selected_model = st.selectbox(
    "Select a model",
    ["DistilBERT", "DistilGPT-2", "Compare Both"],
    label_visibility="collapsed"
)


# --------------------------------------------------
# 9. SENTIMENT ANALYSIS
# --------------------------------------------------

if st.button("Analyze Sentiment  →", use_container_width=True):

    # Check whether the review input is empty
    if not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        # Decide which models will perform the analysis
        if selected_model == "Compare Both":
            models_to_run = ["DistilBERT", "DistilGPT-2"]
        else:
            models_to_run = [selected_model]

        # Create columns for side-by-side comparison
        if len(models_to_run) == 2:
            result_columns = st.columns(2)
        else:
            result_columns = [st.container()]

        # Run sentiment analysis for each selected model
        for index, model_name in enumerate(models_to_run):

            with st.spinner(f"Analyzing with {model_name}..."):

                # Load the selected model
                classifier = load_model(model_name)

                # Predict sentiment from the user's review
                result = classifier(
                    review,
                    truncation=True,
                    max_length=256
                )[0]

            # Extract the model's prediction label
            label = result["label"]

            # Convert the label into readable sentiment
            sentiment = (
                "Positive"
                if label in ["LABEL_1", "1"]
                else "Negative"
            )

            # Calculate confidence as a percentage
            confidence = result["score"] * 100

            # Set a different accent for positive and negative results
            if sentiment == "Positive":
                sentiment_color = "#C5F36A"
                sentiment_icon = "POSITIVE"
            else:
                sentiment_color = "#FF9D52"
                sentiment_icon = "NEGATIVE"

            # Display the result inside its column
            with result_columns[index]:

                st.markdown(f"""
                <div class="result-card">

                    <div class="model-name">
                        {model_name} MODEL
                    </div>

                    <div class="prediction"
                         style="color: {sentiment_color};">
                        {sentiment_icon}
                    </div>

                    <div class="confidence">
                        Prediction confidence: {confidence:.2f}%
                    </div>

                    <div class="progress-track">
                        <div class="progress-fill"
                             style="width: {confidence:.2f}%;
                             background: {sentiment_color};">
                        </div>
                    </div>

                </div>
                """, unsafe_allow_html=True)


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.markdown("""
<hr>

<div class="footer">
    BUILT BY FAREED | UNIVERSITY OF GWADAR<br>
    Encoder-Only vs Decoder-Only Language Models<br>
    Powered by Hugging Face & Streamlit
</div>
""", unsafe_allow_html=True)
