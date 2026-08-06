import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import box


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

#Calcul de l'emprise
bounds = boundary.total_bounds

minx,miny,maxx,maxy = bounds

center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

# Créer la carte
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10
)


# Ajouter fond satellite Esri
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False
).add_to(m)


# Créer un masque autour de la zone d'étude

world = box(-180, -90, 180, 90)

mask = world.difference(boundary.geometry.iloc[0])

mask_gdf = gpd.GeoDataFrame(
    geometry=[mask],
    crs="EPSG:4326"
)

folium.GeoJson(
    mask_gdf,
    style_function=lambda x: {
        "fillColor": "black",
        "color": "black",
        "fillOpacity": 0.5,
        "weight": 0
    }
).add_to(m)



# Ajouter la limite
folium.GeoJson(
    boundary,
    name="Application boundary",
    style_function=lambda x: {
        "fillColor": "transparent",
        "color": "black",
        "weight": 3
    }
).add_to(m)

#Zoom auto sur la zone
m.fit_bounds([
    [miny, minx],
    [maxy, maxx]
])

# Afficher la carte
st_folium(
    m,
    width=1000,
    height=700
)
