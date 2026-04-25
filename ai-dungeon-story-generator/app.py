import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

generator = load_model()

st.set_page_config(page_title="AI Dungeon Story Generator")

st.title("🧙 AI Dungeon Story Generator")
st.markdown(
    """
    Create dynamic AI-powered stories.
    Choose a genre, enter your idea, and explore multiple continuations.
    """
)

st.write("Generate interactive fantasy stories using AI!")

# Genre selection
genre = st.selectbox(
    "Select Genre:",
    ["Fantasy", "Mystery", "Horror", "Sci-Fi"]
)

# User input
user_input = st.text_area(
    "Enter your story idea:",
    placeholder="A young warrior finds a glowing sword in the forest..."
)

# Generate button
generate_button = st.button("Generate Story")
if generate_button:
    if user_input.strip() == "":
        st.warning("Please enter a story idea first.")
    else:
        # generation code here
        prompt = f"In this {genre.lower()} tale, {user_input}"
        
        with st.spinner("Generating story..."):
            outputs = generator(
                prompt,
                max_new_tokens=80,
                num_return_sequences=2,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )
        
        for i, story in enumerate(outputs):
            st.subheader(f"Story {i+1}")
            
            full_text = story["generated_text"]
            
            # Remove prompt part
            cleaned_text = full_text.replace(prompt, "").strip()
            
            st.write(cleaned_text)
            
            st.download_button(
                label=f"Download Story {i+1}",
                data=cleaned_text,
                file_name=f"story_{i+1}.txt",
                mime="text/plain"
            )
            if st.button("Clear Stories", key="clear_button"):
                st.session_state.stories = []