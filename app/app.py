import streamlit as st
import ee
import geemap.foliumap as geemap

# Configuration de la page Streamlit
st.set_page_config(layout="wide")
st.title("Moniteur de Végétation - Yorta Yorta")

# 1. INITIALISATION DE EARTH ENGINE
# Note : Assurez-vous d'avoir configuré vos identifiants EE sur votre serveur/machine
try:
    ee.Initialize()
except Exception as e:
    st.error("Veuillez authentifier Google Earth Engine.")
    # ee.Authenticate() # À exécuter une fois localement si nécessaire

# Définissez votre zone d'étude (Exemple : un point ou un polygone importé)
# Pour l'exemple, nous créons un point autour de la région Yorta Yorta (Australie)
# À remplacer par votre propre variable 'geometry' ou 'roi'
roi = ee.Geometry.Point([145.0, -36.0]).buffer(20000) 

# Périodes
debut_P1 = '2019-01-01'
fin_P1   = '2020-12-31'
debut_P2 = '2024-01-01'
fin_P2   = '2025-12-31'

# Fonction Cloud Mask Sentinel-2
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

# Fonction pour calculer le NDVI sur une image unique
def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# 2. CHARGEMENT DES COLLECTIONS COMPLÈTES
collection_P1 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(debut_P1, fin_P1)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .map(maskS2clouds)
                 .map(addNDVI))

collection_P2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(debut_P2, fin_P2)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .map(maskS2clouds)
                 .map(addNDVI))

# 3. CRÉATION DU MASQUE FORESTIER ULTRA-STRICT (Anti-cultures)
ndvi_std_P1 = collection_P1.select('NDVI').reduce(ee.Reducer.stdDev())
ndvi_median_P1 = collection_P1.select('NDVI').median()
ndvi_min_P1 = collection_P1.select('NDVI').reduce(ee.Reducer.min())

# Application des critères stricts
masqueForetSentinel = (ndvi_std_P1.lt(0.05)
                       .And(ndvi_median_P1.gt(0.70))
                       .And(ndvi_min_P1.gt(0.50))
                       .clip(roi))

# 4. GÉNÉRATION DES COMPOSITES FINAUX
s2_P1_median = collection_P1.median().clip(roi)
s2_P2_median = collection_P2.median().clip(roi)

ndvi_P1_foret = s2_P1_median.select('NDVI').updateMask(masqueForetSentinel)
ndvi_P2_foret = s2_P2_median.select('NDVI').updateMask(masqueForetSentinel)

# 5. CALCUL DU CHANGEMENT REEL EN FORÊT
changementNDVI = ndvi_P2_foret.subtract(ndvi_P1_foret).rename('Changement')

# 6. CONFIGURATION DE L'AFFICHAGE AVEC GEEMAP
Map = geemap.Map()
Map.centerObject(roi, 10)
Map.add_basemap('HYBRID')

# Paramètres de visualisation
vis_ndvi = {'min': 0.5, 'max': 0.85, 'palette': ['#ece7f2', '#a6bddb', '#016450']}
vis_changement = {'min': -0.15, 'max': 0.15, 'palette': ['#d73027', '#ffffff', '#1a9850']}

# Ajout des couches sur la carte
Map.addLayer(s2_P2_median, {'bands': ['B4', 'B3', 'B2'], 'max': 0.3}, '1. Fond de carte Sentinel-2 2024-2025', False)
Map.addLayer(ndvi_P1_foret, vis_ndvi, '2. NDVI Forêt Stable (2019-2020)', False)
Map.addLayer(ndvi_P2_foret, vis_ndvi, '3. NDVI Forêt Évolué (2024-2025)', False)
Map.addLayer(changementNDVI, vis_changement, '4. Déforestation/Gain réel (Hors cultures)')

# Rendu de la carte dans l'interface Streamlit
Map.to_streamlit(height=700)
