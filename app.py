import streamlit as st
import pickle
import string
import warnings
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Initialize Porter Stemmer
ps = PorterStemmer()


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


# Download NLTK data
download_nltk_data()


def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stem the words
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer with error handling
@st.cache_resource
def load_model_and_vectorizer():
    try:
        import sklearn
        current_version = sklearn.__version__

        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))

        # Show version info in sidebar
        st.sidebar.info(f"📊 Model Info\n\nScikit-learn: {current_version}")

        if current_version != '1.4.2':
            st.sidebar.warning(
                "⚠️ Model was trained with different sklearn version. Consider retraining for best results.")

        return tfidf, model, True
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False


# Load the model and vectorizer
tfidf, model, models_loaded = load_model_and_vectorizer()

# Streamlit UI
st.title("📧 Email/SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not spam.")

# Only show the interface if models are loaded successfully
if models_loaded:
    input_sms = st.text_area("Enter the message:", height=100, placeholder="Type your message here...")

    if st.button('🔍 Predict', type="primary"):
        if input_sms.strip():  # Check if input is not empty
            try:
                # 1. Preprocess
                transformed_sms = transform_text(input_sms)

                # 2. Vectorize
                vector_input = tfidf.transform([transformed_sms])

                # 3. Predict
                result = model.predict(vector_input)[0]

                # Get prediction probability for confidence score
                prob = model.predict_proba(vector_input)[0]
                confidence = max(prob) * 100

                # 4. Display results
                col1, col2 = st.columns(2)

                with col1:
                    if result == 1:
                        st.error("🚨 **SPAM**")
                    else:
                        st.success("✅ **NOT SPAM**")

                with col2:
                    st.info(f"Confidence: {confidence:.1f}%")

                # Show processed text (optional - for debugging)
                with st.expander("View processed text"):
                    st.write(f"Original: {input_sms}")
                    st.write(f"Processed: {transformed_sms}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("⚠️ Please enter a message to classify.")
else:
    st.error(
        "❌ Cannot load the required model files. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory as this app.")
    st.info("💡 Make sure you have trained and saved your model first!")

# Add some information about the app
st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. **Text Preprocessing**: Converts text to lowercase, removes punctuation, stopwords, and applies stemming
2. **Vectorization**: Converts processed text into numerical features using TF-IDF
3. **Classification**: Uses a trained machine learning model to predict spam/not spam
4. **Results**: Shows prediction with confidence score
""")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and scikit-learn*")