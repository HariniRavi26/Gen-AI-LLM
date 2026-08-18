import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Generation",
    page_icon="🤖"
)

st.title("🤖 AI Text Generation")
st.write("Enter a sentence and let AI complete it for you!")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

generator = load_model()

prompt = st.text_area(
    "Enter your sentence",
    placeholder="Example: Artificial intelligence is changing the world...",
    height=150
)

max_tokens = st.slider(
    "Maximum new words/tokens",
    min_value=20,
    max_value=50,
    value=30,
    step=5
)

if st.button("🚀 Generate Text"):

    if prompt.strip():

        with st.spinner("Generating text..."):

            result = generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated_text = result[0]["generated_text"]

        st.subheader("✨ Generated Text")
        st.write(generated_text)

    else:

        st.warning("⚠️ Please enter a sentence first.")