import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import yaml
import pydeck as pdk
import matplotlib.pyplot as plt

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

AUDIO_DIR = config["audio_dir"]
MANIFEST_PATH = config["manifest_path"]
FEATURES_DIR = config.get("features_path", "features_sampled")

@st.cache_data
def load_manifest(path):
    return pd.read_csv(path)

@st.cache_data
def load_taxonomy():
    taxonomy_path = "taxonomy.csv"
    if os.path.exists(taxonomy_path):
        return pd.read_csv(taxonomy_path)
    return pd.DataFrame()

# Load data
df = load_manifest(MANIFEST_PATH)
taxonomy = load_taxonomy()
if not taxonomy.empty:
    taxonomy.columns = taxonomy.columns.str.strip()
    st.code(f"🧠 Taxonomy columns: {taxonomy.columns.tolist()}")

# Sidebar filters
st.sidebar.title("🔎 Filter Options")

selected_classes = st.sidebar.multiselect(
    "Select Class",
    options=sorted(df["class_name"].dropna().unique()),
    default=sorted(df["class_name"].dropna().unique())
)
df = df[df["class_name"].isin(selected_classes)]

selected_species = st.sidebar.multiselect(
    "Select Species (Common Name)",
    options=sorted(df["common_name"].dropna().unique()),
    default=None
)
if selected_species:
    df = df[df["common_name"].isin(selected_species)]

st.markdown(f"### 📈 Total Samples: {len(df)} | Total Unique Species: {df['common_name'].nunique()}")

# ----------------------
# 📊 Section 1: Class Distribution
# ----------------------
st.header("📊 Class Distribution")
st.caption("💡 Hover over bars to see class-wise counts of audio samples.")
class_counts = df["class_name"].value_counts().reset_index()
class_counts.columns = ["Class", "Count"]
chart1 = alt.Chart(class_counts).mark_bar().encode(
    x=alt.X("Class:N", sort="-y"),
    y="Count:Q",
    tooltip=["Class", "Count"]
).properties(width=700)
st.altair_chart(chart1)

# ----------------------
# 🧬 Section 2: Species Distribution per Class
# ----------------------
st.header("🧬 Species Distribution per Class")
st.caption("💡 Use this chart to detect class imbalance across species.")
species_class = df.groupby(["class_name", "common_name"]).size().reset_index(name="Count")
chart2 = alt.Chart(species_class).mark_bar().encode(
    y=alt.Y("common_name:N", sort="-x"),
    x="Count:Q",
    color="class_name:N",
    tooltip=["class_name", "common_name", "Count"]
).properties(width=700, height=500)
st.altair_chart(chart2)

# ----------------------
# 🐦 Section 2B: Detailed Aves Species Distribution
# ----------------------
if "Aves" in df["class_name"].unique():
    st.header("🐦 Aves: Detailed Species Distribution")
    aves_df = df[df["class_name"] == "Aves"]
    aves_species = aves_df["common_name"].value_counts().reset_index()
    aves_species.columns = ["Species", "Count"]
    st.markdown(f"Total Aves Species: **{aves_species['Species'].nunique()}**")
    chart_aves = alt.Chart(aves_species).mark_bar().encode(
        y=alt.Y("Species:N", sort="-x"),
        x="Count:Q",
        tooltip=["Species", "Count"]
    ).properties(width=700, height=600)
    st.altair_chart(chart_aves)

# ----------------------
# ⭐ Section 3: Quality Rating Distribution
# ----------------------
st.header("⭐ Data Quality Rating Distribution")
st.caption("💡 Use this to check data quality scores. Ratings 1–5, where 5 is best.")
ratings = df[df["rating"] > 0]
rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
rating_counts.columns = ["Rating", "Count"]
chart3 = alt.Chart(rating_counts).mark_bar().encode(
    x="Rating:O",
    y="Count:Q",
    tooltip=["Rating", "Count"]
).properties(width=700)
st.altair_chart(chart3)

# ----------------------
# 🌍 Section 4: Species Location Map
# ----------------------
st.header("📍 Geolocation of Recordings")
st.caption("💡 Explore where recordings were made. Use zoom/pan for better insight. The blue outline represents the El Silencio Nature Reserve (approximate).")
map_df = df.dropna(subset=["latitude", "longitude"])
import random

# Assign unique color per class
unique_classes = map_df["class_name"].dropna().unique()
color_map = {
    cls: [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 160]
    for cls in unique_classes
}
map_df["color"] = map_df["class_name"].map(lambda cls: color_map.get(cls, [128, 128, 128, 160]))

el_silencio_polygon = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "El Silencio Nature Reserve"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-84.75, 10.3],
                    [-84.74, 10.3],
                    [-84.74, 10.31],
                    [-84.75, 10.31],
                    [-84.75, 10.3]
                ]]
            }
        }
    ]
}

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=map_df["latitude"].mean(),
        longitude=map_df["longitude"].mean(),
        zoom=2,
        pitch=0
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=10000,
            pickable=True
        ),
        pdk.Layer(
            "GeoJsonLayer",
            data=el_silencio_polygon,
            stroked=True,
            filled=False,
            get_line_color="[0, 255, 255]",
            line_width_min_pixels=4
        )
    ],
    tooltip={"text": "{common_name} ({scientific_name})\nClass: {class_name}"}
))

# ----------------------
# 🔉 Section 5: Audio Samples Table
# ----------------------
st.header("🔉 Audio Samples")
st.caption("💡 Play recordings and compare by class/species.")
num_samples = st.slider("Number of Samples", 1, 50, 10)
for _, row in df.head(num_samples).iterrows():
    st.subheader(f"{row['common_name']} ({row['scientific_name']})")
    st.markdown(f"**Class:** {row['class_name']} | **Rating:** {row['rating']} | **File:** {row['filename']}")
    file_path = os.path.join(AUDIO_DIR, row["filename"])
    if os.path.exists(file_path):
        st.audio(open(file_path, 'rb').read())
    else:
        st.warning(f"⚠️ Missing file: {row['filename']}")

# ----------------------
# 🔬 Section 6: Feature Sample Explorer
# ----------------------

st.header("🔬 Feature Sample Explorer")
st.caption("Explore denoised audio, Mel spectrograms, and embedding predictions.")

manifest_file = st.selectbox("Select a manifest CSV file to explore", options=[
    "manifest_train.csv", "manifest_val.csv", "manifest_test.csv"
], key="select_manifest")

manifest_path = os.path.join(FEATURES_DIR, manifest_file)
if os.path.exists(manifest_path):
    manifest_df = pd.read_csv(manifest_path)
    st.success(f"Loaded {len(manifest_df)} entries from {manifest_file}.")

    if "chunk_id" not in manifest_df.columns:
        st.error(f"❌ The selected manifest file '{manifest_file}' does not contain a 'chunk_id' column.")
        st.stop()

    sample_choices = manifest_df["chunk_id"].tolist()
    selected_sample = st.selectbox("Select a sample to explore", sample_choices, key="select_sample")
    row = manifest_df[manifest_df["chunk_id"] == selected_sample].iloc[0]

    # 🎧 Denoised Audio
    st.subheader("🎧 Denoised Audio")
    audio_path = os.path.join(FEATURES_DIR, "denoised", row["audio_path"].lstrip("/"))
    if os.path.exists(audio_path):
        st.audio(open(audio_path, 'rb').read())
    else:
        st.warning(f"⚠️ Missing audio file: {audio_path}")

    # 📈 Mel Spectrogram
    st.subheader("📈 Mel Spectrogram")
    mel_path = os.path.join(FEATURES_DIR, "mel", row["mel_path"].lstrip("/"))
    if os.path.exists(mel_path):
        mel_data = np.load(mel_path, allow_pickle=True)
        if isinstance(mel_data, np.lib.npyio.NpzFile):
            mel_data = mel_data[list(mel_data.files)[0]]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(mel_data, aspect="auto", origin="lower")
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ Missing mel file: {mel_path}")

    # 📊 Embedding Predictions
    st.subheader("📊 Predicted Labels from Embeddings")
    emb_path = os.path.join(FEATURES_DIR, "embeddings", row["emb_path"].lstrip("/"))
    if os.path.exists(emb_path):
        probs_data = np.load(emb_path, allow_pickle=True)
        if isinstance(probs_data, np.lib.npyio.NpzFile):
            probs = probs_data[list(probs_data.files)[0]]
        else:
            probs = probs_data

        species = taxonomy.set_index("primary_label")["common_name"].to_dict()
        labels = [species.get(i, f"Species {i}") for i in range(len(probs.flatten()))]

        min_len = min(len(labels), len(probs.flatten()))
        pred_df = pd.DataFrame({
            "Species": labels[:min_len],
            "Probability": probs.flatten()[:min_len]
        })
        pred_df = pred_df.sort_values("Probability", ascending=False)
        primary = pred_df.iloc[0]
        st.code(f"🔍 Raw Primary Label Index: {pred_df.index[0]}, Species: {primary['Species']}, Probability: {primary['Probability']:.4f}")
        secondary = pred_df[pred_df["Probability"] > 0.5].iloc[1:]

        st.markdown(f"**Primary Prediction:** {primary['Species']} ({primary['Probability']:.2f})")
        st.markdown(f"**Secondary Labels (>0.5):**")
        st.dataframe(secondary.reset_index(drop=True))
    else:
        st.warning(f"⚠️ Missing embedding file: {emb_path}")
