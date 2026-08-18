import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍"
)

st.title("🔍 AI Object Detection")
st.write("Upload an image and let AI detect the objects in it!")

@st.cache_resource
def load_model():
    return pipeline(
        "object-detection",
        model="facebook/detr-resnet-50"
    )

detector = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Objects"):

        with st.spinner("Detecting objects..."):

            results = detector(image)

        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)

        detected_objects = []

        for result in results:

            label = result["label"]
            score = result["score"]
            box = result["box"]

            if score >= 0.5:

                xmin = int(box["xmin"])
                ymin = int(box["ymin"])
                xmax = int(box["xmax"])
                ymax = int(box["ymax"])

                draw.rectangle(
                    [xmin, ymin, xmax, ymax],
                    width=3
                )

                text = f"{label} {score:.2f}"

                draw.text(
                    (xmin, max(0, ymin - 20)),
                    text
                )

                detected_objects.append(
                    f"{label} ({score:.2%})"
                )

        st.subheader("🎯 Detected Objects")

        if detected_objects:

            for obj in detected_objects:
                st.write("•", obj)

            st.subheader("🖼️ Detection Result")
            st.image(
                output_image,
                use_container_width=True
            )

        else:

            st.warning(
                "No objects were detected with sufficient confidence."
            )