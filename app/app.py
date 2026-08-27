import io
import json
import os
import tempfile
import zipfile
import pandas as pd
import streamlit as st
import geopandas as gpd
import folium
import ee
from streamlit_folium import st_folium
from shapely.geometry import shape, box
from folium.plugins import Draw
from google.oauth2 import service_account

# =========================================================================
# CONFIG
# =========================================================================
st.set_page_config(
    page_title="Yorta Yorta Vegetation Monitor",
    page_icon="🌿",
    layout="wide",
)

NDVI_PALETTE = ["7f7f7f", "a6611a", "dfc27d", "f5f5f5", "a6d96a", "1a9850", "006837"]
DIFF_PALETTE = ["67001f", "d73027", "fee08b", "ffffff", "d9ef8b", "1a9850", "00441b"]

# Fenêtre phénologique appliquée chaque année (pic de végétation, comparabilité interannuelle)
MONTH_START, MONTH_END = 5, 9  # mai à septembre

# Sentinel-2 SR, composites sur 2 ans pour lisser la variabilité inter-annuelle
RECENT_START, RECENT_END, RECENT_LABEL = "2024-01-01", "2025-12-31", "2024–2025"
EARLY_START, EARLY_END, EARLY_LABEL = "2019-01-01", "2020-12-31", "2019–2020"

CLOUD_PCT_S2 = 40  # filtre scène large : le masquage fin se fait ensuite via SCL au pixel
BOUNDARY_PATH = "data/raw/app_boundary.geojson"

# Sites culturels : rayon de buffer et seuil de "perte significative"
SITE_BUFFER_METERS = 1000
LOSS_THRESHOLD = -0.1  # dNDVI < -0.1 = perte de végétation jugée significative

# =========================================================================
# EARTH ENGINE INIT (une seule fois par session)
# =========================================================================
@st.cache_resource(show_spinner=False)
def init_ee():
    credentials = service_account.Credentials.from_service_account_info(
        dict(st.secrets["earthengine"]),
        scopes=["https://googleapis.com"],
    )
    ee.Initialize(credentials, project=st.secrets["earthengine"]["project_id"])
    return True

try:
    init_ee()
except Exception as e:
    st.error(f"Earth Engine connection failed: {e}")
    st.stop()

# =========================================================================
# HELPERS
# =========================================================================
def geopandas_to_ee(gdf):
    """Convertit la géométrie d'un GeoDataFrame (1 feature) en ee.Geometry."""
    geojson = json.loads(gdf[["geometry"]].to_json())
    return ee.Geometry(geojson["features"][0]["geometry"])

def geom_signature(geom_dict):
    """Signature stable d'une géométrie GeoJSON (dict) pour détecter un changement."""
    return json.dumps(geom_dict, sort_keys=True)

def shapely_to_ee(geom):
    """Convertit une géométrie shapely (WGS84) en ee.Geometry."""
    geojson = json.loads(gpd.GeoSeries([geom], crs="EPSG:4326").to_json())
    return ee.Geometry(geojson["features"][0]["geometry"])

# ---- Import des sites culturels (KML / SHP zippé / CSV) ----
def _read_kml_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path, driver="KML")
    finally:
        os.unlink(tmp_path)
    return gdf

def _read_shp_zip_bytes(file_bytes):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "sites.zip")
        with open(zip_path, "wb") as f:
            f.write(file_bytes)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp_dir)
        shp_files = [f for f in os.listdir(tmp_dir) if f.lower().endswith(".shp")]
        if not shp_files:
            raise ValueError("No .shp file found inside the zip archive.")
        gdf = gpd.read_file(os.path.join(tmp_dir, shp_files[0])).copy()
    return gdf

def _read_csv_bytes(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = next((cols_lower[c] for c in cols_lower if c in ("lat", "latitude", "y")), None)
    lon_col = next((cols_lower[c] for c in cols_lower if c in ("lon", "long", "longitude", "x")), None)
    if lat_col is None or lon_col is None:
        raise ValueError("CSV must contain latitude/longitude columns (lat/latitude/y and lon/long/longitude/x).")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

@st.cache_data(show_spinner=False)
def parse_sites_file(file_bytes, filename):
    """Parse un fichier de points (KML, SHP zippé ou CSV lat/lon) -> GeoDataFrame de Points en EPSG:4326."""
    ext = filename.lower().split(".")[-1]
    if ext == "kml":
        gdf = _read_kml_bytes(file_bytes)
    elif ext == "zip":
        gdf = _read_shp_zip_bytes(file_bytes)
    elif ext == "csv":
        gdf = _read_csv_bytes(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.empty:
        raise ValueError("No point geometries found in the uploaded file.")
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.reset_index(drop=True)

def site_label_column(gdf):
    """Devine la colonne de nom du site, sinon None."""
    for candidate in ("name", "nom", "site", "label", "Name", "NOM"):
        if candidate in gdf.columns:
            return candidate
    return None

def utm_epsg_for(lon, lat):
    """EPSG UTM adapté à la position (pour un buffer en mètres correct)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

def buffer_points_1km(points_gdf, meters=SITE_BUFFER_METERS):
    """Buffer en mètres autour de chaque point, reprojection UTM automatique."""
    centroid = points_gdf.geometry.unary_union.centroid
    utm_crs = utm_epsg_for(centroid.x, centroid.y)
    points_utm = points_gdf.to_crs(utm_crs)
    buffers_utm = points_utm.copy()
    buffers_utm["geometry"] = points_utm.geometry.buffer(meters)
    return buffers_utm.to_crs("EPSG:4326")

def mask_s2_scl(img):
    """Masque nuages / ombres / cirrus / pixels défectueux via la bande SCL."""
    scl = img.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))  # végétation, sol nu, eau, non classé
    return img.updateMask(good)

def sentinel2_ndvi_image(aoi, start, end, cloud_pct=CLOUD_PCT_S2):
    """Composite NDVI médian Sentinel-2, filtré pour exclure l'agriculture
    en ne gardant que la forêt stable et dense (basse variabilité, haut minimum)."""
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filter(ee.Filter.calendarRange(MONTH_START, MONTH_END, "month"))
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_scl)
    )
    
    n = coll.size().getInfo()
    if n == 0:
        return None, 0
    
    # 1. Calculer le NDVI pour chaque scène de la série temporelle
    def compute_collection_ndvi(img):
        return img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        
    ndvi_collection = coll.map(compute_collection_ndvi)
    
    # 2. Extraire les signatures statistiques d'exclusion
    ndvi_median = ndvi_collection.median()
    ndvi_std = ndvi_collection.reduce(ee.Reducer.stdDev())
    ndvi_min = ndvi_collection.reduce(ee.Reducer.min())
    
    # 3. Masque forestier strict (anti-cultures)
    # stdDev < 0.06 : élimine les cultures (fortes variations à la récolte)
    # median > 0.65 : sélectionne le couvert arboré haut
    # min > 0.45    : bloque les parcelles fauchées, labourées ou récoltées
    masque_foret = (ndvi_std.lt(0.06)
                    .And(ndvi_median.gt(0.65))
                    .And(ndvi_min.gt(0.45)))
    
    # 4. Assigner le masque au composite final
    ndvi_final = ndvi_median.updateMask(masque_foret).rename("NDVI").clip(aoi)
    
    return ndvi_final, n

@st.cache_data(ttl=3600, show_spinner=False)
def sentinel2_ndvi_tile(aoi_geojson_str, start, end, cloud_pct):
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    ndvi, n = sentinel2_ndvi_image(aoi, start, end, cloud_pct)
    if ndvi is None:
        return None, 0
    tile = ndvi.getMapId({"min": -1, "max": 1, "palette": NDVI_PALETTE})
    return tile["tile_fetcher"].url_format, n

def _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end):
    """Image dNDVI (recent - early) Sentinel-2, clippée à aoi. None si pas d'images."""
    early_ndvi, n_early = sentinel2_ndvi_image(aoi, early_start, early_end)
    recent_ndvi, n_recent = sentinel2_ndvi_image(aoi, recent_start, recent_end)
    if early_ndvi is None or recent_ndvi is None:
        return None
    return recent_ndvi.subtract(early_ndvi).rename("dNDVI").clip(aoi)

@st.cache_data(ttl=3600, show_spinner=False)
def ndvi_diff_tile(aoi_geojson_str, early_start, early_end, recent_start, recent_end):
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    diff = _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end)
    if diff is None:
        return None
    tile = diff.getMapId({"min": -0.4, "max": 0.4, "palette": DIFF_PALETTE})
    return tile["tile_fetcher"].url_format

@st.cache_data(ttl=3600, show_spinner=False)
def vegetation_loss_tile(aoi_geojson_str, early_start, early_end, recent_start, recent_end, threshold):
    """Masque rouge : pixels où dNDVI < threshold (perte significative), clippé à aoi."""
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    diff = _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end)
    if diff is None:
        return None
    loss_mask = diff.lt(threshold).selfMask()
    tile = loss_mask.getMapId({"palette": ["ff0000"], "min": 0, "max": 1})
    return tile["tile_fetcher"].url_format

def add_tile_layer(fmap, url, name, show=False):
    folium.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
