import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# ----------------------
# Setup
# ----------------------
st.set_page_config(page_title="Fundbüro Bild-Erkennung", page_icon="🔍")
st.title("🔍 Fundbüro KI Bild-Erkennung")
st.write("Lade ein Bild hoch, und die KI zeigt dir, welchem Fundstück es am ähnlichsten ist.")

# Lade das Modell
@st.cache_resource
def load_ki_model():
    return load_model("keras_Model.h5", compile=False)

model = load_ki_model()

# Lade Labels
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_labels()

# ----------------------
# Bild-Upload
# ----------------------
uploaded_file = st.file_uploader("📷 Wähle ein Bild aus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Zeige das hochgeladene Bild
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    # Bild vorbereiten
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader("Ergebnis der KI")
    st.write(f"**Gefundenes Objekt:** {class_name}")
    st.write(f"**Wahrscheinlichkeit:** {confidence_score:.2%}")
