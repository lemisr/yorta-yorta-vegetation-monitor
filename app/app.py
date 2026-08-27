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

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Yorta Yorta Vegetation Monitor",
    page_icon="🌿",
    layout="wide",
)

NDVI_PALETTE = ["7f7f7f", "a6611a", "dfc27d", "f5f5f5", "a6d96a", "1a9850", "006837"]
DIFF_PALETTE = ["67001f", "d73027", "fee08b", "ffffff", "d9ef8b", "1a9850", "00441b"]
LOSS_COLOR_CYCLE = ["ff0000", "ff6600", "ffcc00"]  # couleurs distinctes en mode comparaison

# Fenêtre phénologique appliquée chaque année (pic de végétation, comparabilité inter-annuelle)
MONTH_START, MONTH_END = 5, 9  # mai à septembre

# Sentinel-2 SR, composites sur 2 ans pour lisser la variabilité inter-annuelle
# (une seule année pourrait être anormalement sèche/humide et fausser la comparaison).
RECENT_START, RECENT_END, RECENT_LABEL = "2024-01-01", "2025-12-31", "2024–2025"
EARLY_START, EARLY_END, EARLY_LABEL = "2019-01-01", "2020-12-31", "2019–2020"

CLOUD_PCT_S2 = 40  # filtre scène large : le masquage fin se fait ensuite via SCL au pixel

BOUNDARY_PATH = "data/raw/app_boundary.geojson"

# Sites culturels : rayon de buffer et seuil de "perte significative"
SITE_BUFFER_METERS = 1000
LOSS_THRESHOLD = -0.1  # dNDVI < -0.1 = perte de végétation jugée significative

# Masque forêt : deux méthodes disponibles.
# - "ndvi" : NDVI médian sur l'ANNEE COMPLETE (pas juste mai-sept) >= seuil. Une forêt
#   reste dense toute l'année ; un champ cultivé non (cycle de culture / sol nu après
#   récolte). Simple mais un seuil seul confond encore certains champs très verts.
# - "dw"   : Google Dynamic World (classification Sentinel-2 déjà entraînée, 10m,
#   quasi temps réel) — probabilité "trees" par pixel >= seuil. Un vrai classifieur
#   au lieu d'un seuil NDVI arbitraire, donc en général moins de faux positifs sur
#   les cultures très vertes.
NDVI_MASK_CHOICES = {
    "Désactivé": [],
    "0.5": [0.5],
    "0.6": [0.6],
    "Comparer 0.5 vs 0.6": [0.5, 0.6],
}


# =========================================================
# EARTH ENGINE INIT (une seule fois par session)
# =========================================================
@st.cache_resource(show_spinner=False)
def init_ee():
    credentials = service_account.Credentials.from_service_account_info(
        dict(st.secrets["earthengine"]),
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )
    ee.Initialize(credentials, project=st.secrets["earthengine"]["project_id"])
    return True


try:
    init_ee()
except Exception as e:
    st.error(f"Earth Engine connection failed: {e}")
    st.stop()


# =========================================================
# HELPERS
# =========================================================
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
    lon_col = next(
        (cols_lower[c] for c in cols_lower if c in ("lon", "long", "longitude", "x")), None
    )
    if lat_col is None or lon_col is None:
        raise ValueError(
            "CSV must contain latitude/longitude columns "
            "(lat/latitude/y and lon/long/longitude/x)."
        )

    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
    )


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
    """Masque nuages / ombres / cirrus / pixels défectueux via la bande SCL (au pixel, pas à la scène)."""
    scl = img.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))  # végétation, sol nu, eau, non classé
    return img.updateMask(good)


def _s2_collection(aoi, start, end, cloud_pct=CLOUD_PCT_S2, month_range=None):
    """Collection Sentinel-2 SR filtrée + masquée SCL. month_range=(m1,m2) optionnel
    pour restreindre à une fenêtre phénologique ; None = année complète."""
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_scl)
    )
    if month_range is not None:
        coll = coll.filter(ee.Filter.calendarRange(month_range[0], month_range[1], "month"))
    return coll


def sentinel2_ndvi_image(aoi, start, end, cloud_pct=CLOUD_PCT_S2):
    """Composite NDVI médian Sentinel-2, fenêtre mai-sept de chaque année (pic de
    végétation, utilisé pour l'affichage NDVI / dNDVI principal)."""
    coll = _s2_collection(aoi, start, end, cloud_pct, month_range=(MONTH_START, MONTH_END))
    n = coll.size().getInfo()
    if n == 0:
        return None, 0
    ndvi = coll.median().normalizedDifference(["B8", "B4"]).rename("NDVI").clip(aoi)
    return ndvi, n


def full_year_ndvi_image(aoi, start, end, cloud_pct=CLOUD_PCT_S2):
    """Composite NDVI médian Sentinel-2 sur l'ANNEE COMPLETE (pas de filtre mensuel).
    Sert uniquement à construire le masque forêt : une forêt garde un NDVI élevé
    toute l'année, contrairement à une culture (cycle de croissance / récolte / sol nu),
    donc ce composite permet de séparer les deux là où un NDVI mai-sept seul ne le peut pas."""
    coll = _s2_collection(aoi, start, end, cloud_pct, month_range=None)
    n = coll.size().getInfo()
    if n == 0:
        return None, 0
    ndvi = coll.median().normalizedDifference(["B8", "B4"]).rename("NDVI").clip(aoi)
    return ndvi, n


def dynamic_world_forest_mask(aoi, prob_threshold, start=EARLY_START, end=EARLY_END):
    """Masque binaire (selfMask) basé sur Google Dynamic World : moyenne de la
    probabilité 'trees' par pixel sur la période, seuillée. DW est déjà un
    classifieur entraîné sur Sentinel-2 (10m), donc plus robuste qu'un simple
    seuil NDVI pour distinguer forêt et culture très verte."""
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(aoi)
        .select("trees")
    )
    n = dw.size().getInfo()
    if n == 0:
        return None
    trees_prob = dw.mean().clip(aoi)
    return trees_prob.gte(prob_threshold).selfMask()


def get_forest_mask(aoi, spec, start=EARLY_START, end=EARLY_END):
    """Masque binaire (selfMask) pour un spec = ("ndvi", seuil) ou ("dw", seuil).
    Appliqué en aval pour exclure les champs cultivés des couches NDVI / dNDVI /
    détection de perte."""
    if spec is None:
        return None
    method, value = spec
    if method == "ndvi":
        ndvi, n = full_year_ndvi_image(aoi, start, end)
        if ndvi is None:
            return None
        return ndvi.gte(value).selfMask()
    elif method == "dw":
        return dynamic_world_forest_mask(aoi, value, start, end)
    return None


def _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end):
    """Image dNDVI (recent - early) Sentinel-2, clippée à aoi. None si pas d'images."""
    early_ndvi, n_early = sentinel2_ndvi_image(aoi, early_start, early_end)
    recent_ndvi, n_recent = sentinel2_ndvi_image(aoi, recent_start, recent_end)
    if early_ndvi is None or recent_ndvi is None:
        return None
    return recent_ndvi.subtract(early_ndvi).rename("dNDVI").clip(aoi)


def _apply_forest_mask(img, aoi, forest_spec):
    """Applique le masque forêt (si un spec est fourni) à une image ee.Image."""
    if forest_spec is None or img is None:
        return img
    mask = get_forest_mask(aoi, forest_spec)
    if mask is None:
        return img
    return img.updateMask(mask)


@st.cache_data(ttl=3600, show_spinner=False)
def sentinel2_ndvi_tile(aoi_geojson_str, start, end, cloud_pct, forest_spec=None):
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    ndvi, n = sentinel2_ndvi_image(aoi, start, end, cloud_pct)
    if ndvi is None:
        return None, 0
    ndvi = _apply_forest_mask(ndvi, aoi, forest_spec)
    tile = ndvi.getMapId({"min": -1, "max": 1, "palette": NDVI_PALETTE})
    return tile["tile_fetcher"].url_format, n


@st.cache_data(ttl=3600, show_spinner=False)
def ndvi_diff_tile(aoi_geojson_str, early_start, early_end, recent_start, recent_end, forest_spec=None):
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    diff = _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end)
    if diff is None:
        return None
    diff = _apply_forest_mask(diff, aoi, forest_spec)
    tile = diff.getMapId({"min": -0.4, "max": 0.4, "palette": DIFF_PALETTE})
    return tile["tile_fetcher"].url_format


@st.cache_data(ttl=3600, show_spinner=False)
def vegetation_loss_tile(
    aoi_geojson_str, early_start, early_end, recent_start, recent_end, threshold, forest_spec=None,
    color="ff0000",
):
    """Masque coloré : pixels où dNDVI < threshold (perte significative), clippé à aoi,
    restreint aux pixels forêt si forest_spec est fourni."""
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    diff = _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end)
    if diff is None:
        return None
    loss_mask = diff.lt(threshold).selfMask()
    loss_mask = _apply_forest_mask(loss_mask, aoi, forest_spec)
    tile = loss_mask.getMapId({"palette": [color], "min": 0, "max": 1})
    return tile["tile_fetcher"].url_format


def add_tile_layer(fmap, url, name, show=False):
    folium.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=show,
    ).add_to(fmap)


def legend_html(title, low_label, high_label, palette, bottom):
    gradient = ", ".join(f"#{c}" for c in palette)
    return f"""
    <div style="position: fixed; bottom: {bottom}px; left: 30px; width: 260px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:11px; padding:7px;">
      <b>{title}</b><br><br>
      <div style="height:20px; width:240px;
                  background: linear-gradient(to right, {gradient});"></div>
      <div style="display:flex; justify-content:space-between; font-size:12px;">
        <span>{low_label}</span><span>{high_label}</span>
      </div>
    </div>
    """


# =========================================================
# ETAT
# =========================================================
if "drawn_geom" not in st.session_state:
    st.session_state.drawn_geom = None  # dict GeoJSON, validé et dans la limite
if "drawn_invalid_msg" not in st.session_state:
    st.session_state.drawn_invalid_msg = None


# =========================================================
# UI - TITRE / UPLOAD
# =========================================================
st.title("🌿 Yorta Yorta Vegetation Monitor")
st.write("Interactive vegetation monitoring application using GIS and remote sensing.")

uploaded_file = st.file_uploader("Upload your analysis area (GeoJSON)", type=["geojson"])

st.subheader("🏛️ Cultural / heritage sites")
uploaded_sites_file = st.file_uploader(
    "Upload site locations (KML, zipped Shapefile, or CSV with lat/lon columns)",
    type=["kml", "zip", "csv"],
)

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("Clear drawn selection"):
        st.session_state.drawn_geom = None
        st.session_state.drawn_invalid_msg = None
        st.rerun()

# ---- Masque forêt : contrôle sidebar ----
st.sidebar.subheader("🌲 Forest mask")
st.sidebar.caption(
    "Excludes cropland so vegetation-loss layers only show forest, not "
    "harvested/rotated fields."
)
mask_method = st.sidebar.radio(
    "Method",
    options=["NDVI (annual)", "Dynamic World"],
    index=1,  # on teste Dynamic World par défaut
)

if mask_method == "NDVI (annual)":
    st.sidebar.caption(
        f"Keeps pixels whose full-year median NDVI ({EARLY_LABEL}) stays above "
        "the threshold — forest stays dense year-round, crops don't."
    )
    ndvi_mode = st.sidebar.radio(
        "NDVI threshold",
        options=list(NDVI_MASK_CHOICES.keys()),
        index=1,  # "0.5"
    )
    active_forest_specs = [("ndvi", t) if t is not None else None for t in (NDVI_MASK_CHOICES[ndvi_mode] or [None])]
else:
    st.sidebar.caption(
        "Google Dynamic World: pixel-level tree probability from a trained "
        f"Sentinel-2 classifier, averaged over {EARLY_LABEL}."
    )
    dw_threshold = st.sidebar.slider("Tree probability threshold", 0.0, 1.0, 0.5, 0.05)
    active_forest_specs = [("dw", dw_threshold)]

# Limite de l'application (rectangle déjà présent dans le repo)
boundary = gpd.read_file(BOUNDARY_PATH)
boundary_geom = boundary.geometry.iloc[0]


# =========================================================
# DETERMINER L'AOI ACTIVE (upload prioritaire, sinon dessin validé)
# =========================================================
active_gdf = None
active_source = None  # "upload" | "draw"

if uploaded_file is not None:
    uploaded_area = gpd.read_file(uploaded_file)
    if uploaded_area.crs != boundary.crs:
        uploaded_area = uploaded_area.to_crs(boundary.crs)

    if uploaded_area.geometry.iloc[0].within(boundary_geom):
        active_gdf = uploaded_area
        active_source = "upload"
        st.success("✅ GeoJSON loaded and inside application boundary")
    else:
        st.error("❌ Uploaded area is outside the application boundary — ignored.")

elif st.session_state.drawn_geom is not None:
    drawn_gdf = gpd.GeoDataFrame(
        geometry=[shape(st.session_state.drawn_geom)], crs="EPSG:4326"
    )
    if drawn_gdf.crs != boundary.crs:
        drawn_gdf = drawn_gdf.to_crs(boundary.crs)
    active_gdf = drawn_gdf
    active_source = "draw"

if st.session_state.drawn_invalid_msg:
    st.error(st.session_state.drawn_invalid_msg)


# =========================================================
# CALCUL NDVI (si une AOI active et valide existe)
# Une entrée par seuil forêt actif (None = pas de masque, ou [0.5], [0.6], [0.5, 0.6])
# =========================================================
ndvi_layers = []  # liste de dicts : threshold, tile_recent, tile_early, tile_diff, n_recent, n_early
n_recent = n_early = 0

if active_gdf is not None:
    aoi_ee = geopandas_to_ee(active_gdf)
    aoi_geojson_str = json.dumps(aoi_ee.getInfo())

    try:
        for spec in active_forest_specs:
            tile_recent, n_recent = sentinel2_ndvi_tile(
                aoi_geojson_str, RECENT_START, RECENT_END, CLOUD_PCT_S2, forest_spec=spec
            )
            tile_early, n_early = sentinel2_ndvi_tile(
                aoi_geojson_str, EARLY_START, EARLY_END, CLOUD_PCT_S2, forest_spec=spec
            )
            tile_diff = ndvi_diff_tile(
                aoi_geojson_str, EARLY_START, EARLY_END, RECENT_START, RECENT_END, forest_spec=spec
            )
            ndvi_layers.append({
                "spec": spec,
                "tile_recent": tile_recent,
                "tile_early": tile_early,
                "tile_diff": tile_diff,
            })
    except Exception as e:
        st.error(f"Earth Engine computation failed: {e}")

    st.caption(
        f"Sentinel-2 images ({RECENT_LABEL}, mai–sept.): {n_recent} | "
        f"Sentinel-2 images ({EARLY_LABEL}, mai–sept.): {n_early}"
    )


# =========================================================
# SITES CULTURELS : parsing, buffer 1km, masque de perte NDVI
# =========================================================
sites_gdf = None
buffers_gdf = None
loss_layers = []  # liste de dicts : threshold, tile_loss
site_name_col = None

if uploaded_sites_file is not None:
    try:
        raw_gdf = parse_sites_file(
            uploaded_sites_file.getvalue(), uploaded_sites_file.name
        )

        # On ne garde que les sites dans la limite de l'application (évite tout crash EE)
        inside_mask = raw_gdf.within(boundary_geom)
        excluded = int((~inside_mask).sum())
        sites_gdf = raw_gdf[inside_mask].reset_index(drop=True)

        if excluded:
            st.warning(f"{excluded} site(s) outside the application boundary were ignored.")

        if sites_gdf.empty:
            st.error("❌ No site is inside the application boundary.")
            sites_gdf = None
        else:
            site_name_col = site_label_column(sites_gdf)
            buffers_gdf = buffer_points_1km(sites_gdf)
            st.success(f"✅ {len(sites_gdf)} site(s) loaded, {SITE_BUFFER_METERS} m buffer applied")

    except Exception as e:
        st.error(f"Could not read the sites file: {e}")

if buffers_gdf is not None:
    try:
        buffer_union_ee = shapely_to_ee(buffers_gdf.geometry.unary_union)
        buffer_geojson_str = json.dumps(buffer_union_ee.getInfo())
        any_loss = False
        for i, spec in enumerate(active_forest_specs):
            color = LOSS_COLOR_CYCLE[i % len(LOSS_COLOR_CYCLE)]
            tile_loss = vegetation_loss_tile(
                buffer_geojson_str, EARLY_START, EARLY_END, RECENT_START, RECENT_END,
                LOSS_THRESHOLD, forest_spec=spec, color=color,
            )
            loss_layers.append({"spec": spec, "tile_loss": tile_loss, "color": color})
            any_loss = any_loss or (tile_loss is not None)
        if not any_loss:
            st.warning("Not enough cloud-free Sentinel-2 images to assess vegetation loss around these sites.")
    except Exception as e:
        st.error(f"Earth Engine computation failed for site buffers: {e}")


# =========================================================
# CONSTRUCTION DE LA CARTE (une seule fois)
# =========================================================
bounds = boundary.total_bounds
minx, miny, maxx, maxy = bounds
center_lat, center_lon = (miny + maxy) / 2, (minx + maxx) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Masque hors de la zone d'étude
world = box(-180, -90, 180, 90)
mask_gdf = gpd.GeoDataFrame(geometry=[world.difference(boundary_geom)], crs="EPSG:4326")
folium.GeoJson(
    mask_gdf,
    style_function=lambda x: {"fillColor": "black", "color": "black", "fillOpacity": 0.5, "weight": 0},
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False,
).add_to(m)

folium.GeoJson(
    boundary,
    name="Application boundary",
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 1},
).add_to(m)

if active_gdf is not None:
    folium.GeoJson(
        active_gdf[["geometry"]].to_json(),
        name="Selected area",
        style_function=lambda x: {"color": "blue", "weight": 2, "fillOpacity": 0},
    ).add_to(m)


def _suffix(spec):
    if spec is None:
        return ""
    method, value = spec
    label = "NDVI" if method == "ndvi" else "DW trees"
    return f" — {label}≥{value}"


for i, layer in enumerate(ndvi_layers):
    spec = layer["spec"]
    show_default = i == 0  # seule la première combinaison est visible par défaut
    if layer["tile_recent"]:
        add_tile_layer(m, layer["tile_recent"], f"NDVI {RECENT_LABEL}{_suffix(spec)} (Sentinel-2)", show=show_default)
    if layer["tile_early"]:
        add_tile_layer(m, layer["tile_early"], f"NDVI {EARLY_LABEL}{_suffix(spec)} (Sentinel-2)", show=False)
    if layer["tile_diff"]:
        add_tile_layer(m, layer["tile_diff"], f"Vegetation change {EARLY_LABEL} → {RECENT_LABEL}{_suffix(spec)}", show=False)

if buffers_gdf is not None:
    folium.GeoJson(
        buffers_gdf[["geometry"]].to_json(),
        name=f"Site buffers ({SITE_BUFFER_METERS} m)",
        style_function=lambda x: {
            "color": "orange",
            "weight": 2,
            "dashArray": "4",
            "fillOpacity": 0,
        },
    ).add_to(m)

    for i, row in sites_gdf.iterrows():
        label = str(row[site_name_col]) if site_name_col else f"Site {i + 1}"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="orange",
            fill=True,
            fill_color="orange",
            fill_opacity=1,
            popup=label,
            tooltip=label,
        ).add_to(m)

for layer in loss_layers:
    if layer["tile_loss"]:
        add_tile_layer(
            m, layer["tile_loss"],
            f"Vegetation loss near sites (dNDVI < {LOSS_THRESHOLD}{_suffix(layer['spec'])})",
            show=True,
        )

draw = Draw(
    export=True,
    draw_options={
        "polyline": False,
        "polygon": True,
        "rectangle": True,
        "circle": False,
        "marker": False,
        "circlemarker": False,
    },
)
draw.add_to(m)

any_ee_layer = any(l["tile_recent"] or l["tile_early"] or l["tile_diff"] for l in ndvi_layers) or any(
    l["tile_loss"] for l in loss_layers
)
if any_ee_layer:
    folium.LayerControl(collapsed=False).add_to(m)
else:
    folium.LayerControl().add_to(m)

m.fit_bounds([[miny, minx], [maxy, maxx]])

if ndvi_layers and ndvi_layers[0]["tile_recent"]:
    legend_title = f"NDVI {RECENT_LABEL}" + _suffix(ndvi_layers[0]["spec"])
    m.get_root().html.add_child(folium.Element(
        legend_html(legend_title, "Low", "High", NDVI_PALETTE, bottom=30)
    ))
if ndvi_layers and any(l["tile_diff"] for l in ndvi_layers):
    m.get_root().html.add_child(folium.Element(
        legend_html(f"Change {EARLY_LABEL}→{RECENT_LABEL}", "Loss", "Gain", DIFF_PALETTE, bottom=140)
    ))
if loss_layers and any(l["tile_loss"] for l in loss_layers):
    rows = "".join(
        f'<span style="display:inline-block; width:14px; height:14px; '
        f'background:#{l["color"]}; margin-right:6px;"></span>'
        f'dNDVI &lt; {LOSS_THRESHOLD}{_suffix(l["spec"])}<br>'
        for l in loss_layers if l["tile_loss"]
    )
    loss_legend = f"""
    <div style="position: fixed; bottom: 250px; left: 30px; width: 260px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:11px; padding:7px;">
      <b>Sites — vegetation loss</b><br><br>
      {rows}
      <span style="font-size:10px;">within {SITE_BUFFER_METERS} m buffer</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(loss_legend))


# =========================================================
# AFFICHAGE (un seul appel st_folium) + capture du dessin
# =========================================================
map_data = st_folium(m, width=1000, height=700, key="main_map")

if map_data and map_data.get("all_drawings"):
    latest = map_data["all_drawings"][-1]
    latest_geom = latest["geometry"]

    already_known = (
        st.session_state.drawn_geom is not None
        and geom_signature(latest_geom) == geom_signature(st.session_state.drawn_geom)
    )

    if not already_known:
        candidate = shape(latest_geom)
        candidate_gdf = gpd.GeoDataFrame(geometry=[candidate], crs="EPSG:4326")
        if candidate_gdf.crs != boundary.crs:
            candidate_gdf = candidate_gdf.to_crs(boundary.crs)

        if candidate_gdf.geometry.iloc[0].within(boundary_geom):
            st.session_state.drawn_geom = latest_geom
            st.session_state.drawn_invalid_msg = None
        else:
            st.session_state.drawn_invalid_msg = (
                "❌ Drawn area is outside the application boundary — not processed."
            )
            st.session_state.drawn_geom = None

        st.rerun()
