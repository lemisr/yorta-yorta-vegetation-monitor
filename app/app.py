import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium


# Configuration
st.set_page_config(
    page_title="Yorta Yorta Vegetation Monitor",
    page_icon="🌿",
    layout="wide"
)


# Titre
st.title("🌿 Yorta Yorta Vegetation Monitor")

st.write(
    "Interactive vegetation monitoring application using GIS and remote sensing."
)


# Charger la limite de l'application
boundary = gpd.read_file(
    "data/raw/app_boundary.geojson"
)


# Créer la carte
m = folium.Map(
    location=[-36.2, 145.2],
    zoom_start=8
)


# Ajouter fond satellite Esri
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False
).add_to(m)


# Ajouter la limite
folium.GeoJson(
    boundary,
    name="Application boundary",
    style_function=lambda x: {
        "fillColor": "transparent",
        "color": "red",
        "weight": 3
    }
).add_to(m)


# Afficher la carte
st_folium(
    m,
    width=1000,
    height=700
)
