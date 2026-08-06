import streamlit as st
import geopandas as gpd
import folium
import ee
from streamlit_folium import st_folium
from shapely.geometry import shape
from shapely.geometry import box
from folium.plugins import Draw

#Connexion GEE
try:
    import json
    from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["earthengine"]),
    scopes=[
        "https://www.googleapis.com/auth/earthengine"
    ]
)

ee.Initialize(
    credentials,
    project=st.secrets["earthengine"]["project_id"]
)

    ee.Initialize(
        credentials,
        project=st.secrets["earthengine"]["project_id"]
    )

    st.success("✅ Earth Engine connected")

except Exception as e:
    st.error(f"Earth Engine connection failed: {e}")

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


# Upload GeoJSON
uploaded_file = st.file_uploader(
    "Upload your analysis area (GeoJSON)",
    type=["geojson"]
)


# Charger la limite de l'application
boundary = gpd.read_file(
    "data/raw/app_boundary.geojson"
)


# Charger la zone utilisateur
uploaded_area = None

if uploaded_file is not None:

    uploaded_area = gpd.read_file(uploaded_file)

    # Harmoniser les projections
    if uploaded_area.crs != boundary.crs:
        uploaded_area = uploaded_area.to_crs(boundary.crs)

    st.success("GeoJSON loaded")


# Calcul de l'emprise
bounds = boundary.total_bounds

minx, miny, maxx, maxy = bounds

center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2



# Créer la carte
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10
)

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

# Ajouter fond satellite Esri
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False
).add_to(m)



# Ajouter limite application
folium.GeoJson(
    boundary,
    name="Application boundary",
    style_function=lambda x: {
        "fillColor": "transparent",
        "color": "black",
        "weight": 1
    }
).add_to(m)



# Ajouter GeoJSON importé
if uploaded_area is not None:

    folium.GeoJson(
        uploaded_area,
        name="Uploaded area",
        style_function=lambda x: {
            "fillColor": "green",
            "color": "green",
            "weight": 1,
            "fillOpacity": 0.3
        }
    ).add_to(m)


# Outil de dessin
draw = Draw(
    export=True,
    draw_options={
        "polyline": False,
        "polygon": True,
        "rectangle": True,
        "circle": False,
        "marker": False,
        "circlemarker": False
    }
)

draw.add_to(m)


# Zoom automatique
m.fit_bounds([
    [miny, minx],
    [maxy, maxx]
])


# Afficher la carte
map_data = st_folium(
    m,
    width=1000,
    height=700
)


# Récupération de la zone dessinée
if map_data and map_data["all_drawings"]:

    drawn = map_data["all_drawings"][-1]

    # Transformer GeoJSON en géométrie
    selected_geometry = shape(drawn["geometry"])

    # Créer une couche GeoDataFrame
    selected_area = gpd.GeoDataFrame(
        geometry=[selected_geometry],
        crs="EPSG:4326"
    )


    # Vérifier projection
    if selected_area.crs != boundary.crs:
        selected_area = selected_area.to_crs(boundary.crs)


    # Validation dans la limite
    is_valid = selected_area.geometry.iloc[0].within(
        boundary.geometry.iloc[0]
    )


    if is_valid:

        st.success("✅ Analysis area validated")

        folium.GeoJson(
            selected_area,
            name="Selected area",
            style_function=lambda x: {
                "fillColor": "blue",
                "color": "blue",
                "weight": 1,
                "fillOpacity": 0.3
            }
        ).add_to(m)

    else:

        st.error("❌ Selected area is outside the application boundary")
