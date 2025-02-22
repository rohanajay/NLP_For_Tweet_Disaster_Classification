import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ============================================================================
# 1. Load the Trained Model and Tokenizer
# ============================================================================

# Path to the saved model directory
model_dir = "./results_fold_2"

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Set model to evaluation mode
model.eval()

# Check if MPS (Apple Silicon GPU) or CPU should be used
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# ============================================================================
# 2. Define Prediction Function
# ============================================================================

def predict_disaster(text):
    """
    Predict whether the input text is about a disaster or not.
    """
    # Tokenize input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move tensors to the appropriate device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, axis=1).item()
    
    # Return prediction label
    return "Disaster" if prediction == 1 else "Not a Disaster"

# ============================================================================
# 3. Build Streamlit App Interface
# ============================================================================

# Set up the Streamlit app structure and title
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")
st.title("🌪️ Disaster Tweet Classifier 🌪️")
st.write("This app predicts whether a given tweet is about a disaster or not.")

# Input text box for user input
user_input = st.text_area("Enter a tweet:", placeholder="Type your tweet here...")

# Button to trigger prediction
if st.button("Classify"):
    if user_input.strip():  # Ensure input is not empty
        result = predict_disaster(user_input)
        
        # Display results with styling and visuals
        if result == "Disaster":
            st.markdown(
                f"""
                <h2 style='color: red; text-align: center; animation: flash 1s infinite;'>🚨 {result}! 🚨</h2>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                "https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", 
                caption="This looks serious!", 
                use_column_width=True,
            )
        else:
            st.markdown(
                f"""
                <h2 style='color: green; text-align: center;'>✅ {result}! ✅</h2>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGZ4MnczZzRqamU3bmlkZ2ExZjZ6NXRsc3hvY2lmdHprcWUycGo3dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/o3HxTpyzyh2XwqCJfa/giphy.gif",
                caption="All clear!", 
                use_column_width=True,
            )
    else:
        st.warning("Please enter some text before classifying.")

# ============================================================================
# 4. Add Custom CSS for Flashing Effect (Disaster Warning)
# ============================================================================

st.markdown(
    """
    <style>
    @keyframes flash {
        0% {opacity: 1;}
        50% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)
