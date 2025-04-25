import os
import logging
import numpy as np
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv()

# Configuration
config = {
    "llama_api_url": "https://api.groq.com/openai/v1/chat/completions",
    "llama_vision_api_url": "https://api.groq.com/openai/v1/chat/completions",
    "api_key": os.getenv("GROQ_API_KEY"),
    "vision_model": "llama-3.2-11b-vision-preview"
}

# Set up logging
logging.basicConfig(filename="eco_chatbot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to get ideas from the Llama model
def get_llama_ideas(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.2-90b-text-preview",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(config["llama_api_url"], headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        logging.info("Response received from Llama model.")
        return response_data['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error getting ideas from Llama model: {e}")
        return "I apologize, but I'm having trouble generating ideas at the moment. Let's try a different approach!"

# Function to generate a response
def generate_response(user_input: str) -> dict:
    prompt = f"""You are EcoBot, an advanced AI assistant specializing in waste management, recycling, and sustainability. Your knowledge encompasses environmental impact, financial aspects, and innovative upcycling ideas. Respond to the user's input: "{user_input}"

If the query is about waste recycling or upcycling:
1. Provide creative and practical ideas, including potential profitability.
2. Estimate the carbon footprint reduction for implementing these ideas.
3. Suggest local resources or businesses that might be interested in the recycled materials.

If the user is making small talk or asking about unrelated topics, respond friendly and steer the conversation back to eco-friendly topics.

Always maintain a positive, encouraging tone. Educate the user about the importance of recycling, waste reduction, and their environmental and economic impacts.

If relevant, mention global recycling trends or successful case studies to inspire the user."""

    response = get_llama_ideas(prompt)
    return {"response": response, "sources": []}  # Simplified for this example

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Classify waste function
def classify_waste(image):
    try:
        # Save the uploaded image temporarily
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_path = "temp_image.png"
            with open(image_path, "wb") as f:
                f.write(output.getvalue())

        # Encode the image
        base64_image = encode_image(image_path)

        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config["vision_model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify this image into one of these waste types: Plastic, Paper, Glass, Metal, Organic, Electronic. Provide a brief explanation for your classification."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(config["llama_vision_api_url"], headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract the classification from the model's response
        classification = result['choices'][0]['message']['content']
        waste_types = ['Plastic', 'Paper', 'Glass', 'Metal', 'Organic', 'Electronic']
        detected_type = next((wt for wt in waste_types if wt.lower() in classification.lower()), "Unknown")
        
        # Simple confidence score (you might want to implement a more sophisticated method)
        confidence = 0.8 if detected_type != "Unknown" else 0.5

        # Clean up the temporary file
        os.remove(image_path)

        return detected_type, confidence, classification

    except Exception as e:
        logging.error(f"Error in waste classification: {e}")
        return "Unknown", 0, f"Error in classification: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="🌿 EcoBot: Ultimate Recycling Assistant", layout="wide")

# Custom CSS for a more polished look
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 20px;
        border: 1px solid #4CAF50;
        padding: 10px 20px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .css-1v0mbdj.ebxwdo61 {
        border-radius: 20px;
        border: 1px solid #4CAF50;
        padding: 20px;
        background-color: #2d2d2d;
    }
    /* Separate chat section */
    .chat-container {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #2d2d2d;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #3d3d3d;
        color: #ffffff;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# Main chat area


# Sidebar
with st.sidebar:
    st.image("./image.png", width=200)
    st.header("🌍 EcoBot Settings")
    user_name = st.text_input("Your Name", "Eco Warrior")
    user_location = st.text_input("Your Location", "New York, NY")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    st.markdown("---")
    st.subheader("🌟 Your Eco Stats")
    st.metric("Carbon Saved", "50 kg", "↑10%")
    st.metric("Items Recycled", "120", "↑5%")
    st.progress(75, text="Eco Score: 75%")

# Main chat area
st.title("🌿 EcoBot: Ultimate Recycling Assistant")
# Create a container for the chat section
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    welcome_message = f"Hello {user_name}! I'm EcoBot, your ultimate recycling assistant. I can help you with creative recycling ideas, financial insights, carbon footprint calculations, and much more. What would you like to explore today?"
    st.session_state.chat_history.append({"role": "assistant", "content": welcome_message, "liked": False, "disliked": False})

# Display chat history with copy, like, and dislike options
for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
            with col1:
                if st.button("📋 Copy", key=f"copy_{i}"):
                    st.toast("Message copied to clipboard!")
            with col2:
                if st.button("👍" if not message["liked"] else "👍 Liked", key=f"like_{i}"):
                    message["liked"] = not message["liked"]
                    message["disliked"] = False
                    st.experimental_rerun()
            with col3:
                if st.button("👎" if not message["disliked"] else "👎 Disliked", key=f"dislike_{i}"):
                    message["disliked"] = not message["disliked"]
                    message["liked"] = False
                    st.experimental_rerun()
            with col4:
                if st.button("🔄 Regenerate", key=f"regen_{i}"):
                    with st.spinner("Regenerating response..."):
                        new_response = generate_response(st.session_state.chat_history[i-1]["content"])
                        message["content"] = new_response["response"]
                        message["liked"] = False
                        message["disliked"] = False
                    st.experimental_rerun()

# Chat input
user_input = st.chat_input("Ask me anything about recycling, upcycling, or sustainability!")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = generate_response(user_input)
            st.markdown(response_data["response"])
            st.session_state.chat_history.append({"role": "assistant", "content": response_data["response"], "liked": False, "disliked": False})
st.markdown('</div>', unsafe_allow_html=True)

# Tabs for additional features
tab1, tab2, tab3 = st.tabs(["📊 Eco-Dashboard", "🔍 Waste Classifier", "🛒 Eco-Shop"])

with tab1:
    st.header("📊 Eco-Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recycling Market Trends")
        # Simulated market data
        materials = ['PET Plastic', 'HDPE Plastic', 'Mixed Paper', 'Aluminum Cans', 'Glass']
        prices = [0.17, 0.57, 0.02, 0.50, -0.02]
        trends = [round(np.random.uniform(-0.05, 0.05), 3) for _ in range(len(materials))]
        market_data = pd.DataFrame({'Material': materials, 'Price ($/lb)': prices, 'Daily Change': trends})
        
        fig = px.bar(market_data, x='Material', y='Price ($/lb)', color='Daily Change',
                     labels={'Price ($/lb)': 'Price per Pound (USD)'},
                     title='Recycling Material Prices')
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏭 Carbon Footprint Calculator")
        waste_type = st.selectbox("Select waste type", market_data['Material'].tolist())
        waste_amount = st.number_input("Enter amount (lbs)", min_value=0.1, value=1.0, step=0.1)
        if st.button("Calculate"):
            footprint = waste_amount * 2.5  # Simplified calculation
            st.metric("Carbon Footprint", f"{footprint:.2f} kg CO2e")
            st.info(f"By recycling {waste_amount} lbs of {waste_type}, you can save approximately {footprint:.2f} kg of CO2 equivalent!")

with tab2:
    st.header("🔍 Waste Classifier")
    st.write("Upload an image of waste, and I'll try to classify it!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        with st.spinner("Classifying..."):
            waste_type, confidence, details = classify_waste(image)
        st.success(f"Detected waste type: {waste_type}")
        st.info(f"Confidence: {confidence:.2f}")
        st.write(f"Details: {details}")
        
        recycling_instructions = {
            "Plastic": "Rinse and recycle in your local plastic recycling bin. Check for recycling numbers.",
            "Paper": "Recycle in your paper recycling bin. Remove any non-paper attachments.",
            "Glass": "Rinse and recycle in your glass recycling bin. Remove caps and lids.",
            "Metal": "Rinse and recycle in your metal recycling bin. Check if your local center accepts the specific type of metal.",
            "Organic": "Compost if possible, or dispose in your organic waste bin.",
            "Electronic": "Take to a specialized e-waste recycling center. Do not dispose in regular trash."
        }
        st.write("Recycling Instructions:", recycling_instructions.get(waste_type, "Please check with your local recycling guidelines for proper disposal."))

with tab3:
    st.header("🛒 Eco-Shop")
    st.write("Discover eco-friendly products that can help reduce your environmental impact!")
    
    # Simulated product data
    products = [
        {"name": "Bamboo Toothbrush Set", "price": "$12.99", "impact": "Reduces plastic waste by 4.7 billion brushes annually"},
        {"name": "Reusable Produce Bags", "price": "$15.99", "impact": "Eliminates need for 1000+ plastic bags over its lifetime"},
        {"name": "Stainless Steel Water Bottle", "price": "$24.99", "impact": "Prevents 167 plastic bottles from landfills yearly"}
    ]
    
    for product in products:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://via.placeholder.com/150", width=150)
        with col2:
            st.subheader(product['name'])
            st.write(f"Price: {product['price']}")
            st.write(f"Environmental Impact: {product['impact']}")
            if st.button(f"Learn More about {product['name']}"):
                st.info("This would link to a detailed product page in a real e-commerce setup.")

# Footer
st.markdown("---")
st.write("🌿 EcoBot - Powered by AI for a Greener Future")
st.write("Remember: Every small action counts towards a sustainable planet!")