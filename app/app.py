import streamlit as st
import ee
import folium
from streamlit_folium import st_folium

# Configuration de la page Streamlit
st.set_page_config(layout="wide")
st.title("Moniteur de Végétation - Yorta Yorta")

# 1. INITIALISATION DE EARTH ENGINE
try:
    ee.Initialize()
except Exception as e:
    st.error("Veuillez authentifier Google Earth Engine.")

# Définissez votre zone d'étude (Région Yorta Yorta, Australie)
roi = ee.Geometry.Point([145.0, -36.0]).buffer(20000) 

# Périodes
debut_P1, fin_P1 = '2019-01-01', '2020-12-31'
debut_P2, fin_P2 = '2024-01-01', '2025-12-31'

# Fonction Cloud Mask Sentinel-2
def maskS2clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

# Fonction pour calculer le NDVI
def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# 2. CHARGEMENT DES COLLECTIONS
collection_P1 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi).filterDate(debut_P1, fin_P1)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .map(maskS2clouds).map(addNDVI))

collection_P2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi).filterDate(debut_P2, fin_P2)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .map(maskS2clouds).map(addNDVI))

# 3. CRÉATION DU MASQUE FORESTIER ULTRA-STRICT
ndvi_std_P1 = collection_P1.select('NDVI').reduce(ee.Reducer.stdDev())
ndvi_median_P1 = collection_P1.select('NDVI').median()
ndvi_min_P1 = collection_P1.select('NDVI').reduce(ee.Reducer.min())

masqueForetSentinel = (ndvi_std_P1.lt(0.05)
                       .And(ndvi_median_P1.gt(0.70))
                       .And(ndvi_min_P1.gt(0.50)).clip(roi))

# 4. GÉNÉRATION DES COMPOSITES
s2_P2_median = collection_P2.median().clip(roi)
ndvi_P1_foret = collection_P1.median().clip(roi).select('NDVI').updateMask(masqueForetSentinel)
ndvi_P2_foret = s2_P2_median.select('NDVI').updateMask(masqueForetSentinel)

# 5. CALCUL DU CHANGEMENT
changementNDVI = ndvi_P2_foret.subtract(ndvi_P1_foret).rename('Changement')

# 6. CONFIGURATION DE LA CARTE FOLIUM NATIVE
# Coordonnées pour centrer la carte (Yorta Yorta)
m = folium.Map(location=[-36.0, 145.0], zoom_start=10, tiles="OpenStreetMap")

# Ajouter un fond de carte Satellite (Google Hybrid) sans passer par geemap
folium.TileLayer(
    tiles="https://google.com{x}&y={y}&z={z}",
    attr="Google",
    name="Google Satellite (Hybrid)",
    overlay=False,
    control=True
).add_to(m)

# Fonction pour obtenir l'URL de la couche Earth Engine
def add_ee_layer(ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(m)

# Paramètres de visualisation
vis_ndvi = {'min': 0.5, 'max': 0.85, 'palette': ['#ece7f2', '#a6bddb', '#016450']}
vis_changement = {'min': -0.15, 'max': 0.15, 'palette': ['#d73027', '#ffffff', '#1a9850']}

# Ajout des couches Earth Engine sur la carte Folium
add_ee_layer(s2_P2_median, {'bands': ['B4', 'B3', 'B2'], 'max': 0.3}, '1. Fond Sentinel-2 2024-2025')
add_ee_layer(ndvi_P1_foret, vis_ndvi, '2. NDVI Forêt Stable (2019-2020)')
add_ee_layer(ndvi_P2_foret, vis_ndvi, '3. NDVI Forêt Évolué (2024-2025)')
add_ee_layer(changementNDVI, vis_changement, '4. Déforestation/Gain réel')

# Ajouter le sélecteur de couches
folium.LayerControl().add_to(m)

# Rendu dans Streamlit
st_folium(m, width=1100, height=700)
