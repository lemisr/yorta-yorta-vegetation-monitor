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

# Période "récente" -> Sentinel-2 SR (bonne couverture / qualité)
RECENT_START, RECENT_END, RECENT_LABEL = "2025-01-01", "2025-12-31", "2025"

# Période "historique" -> Sentinel-2 SR aussi.
# 2017 est la première année où COPERNICUS/S2_SR_HARMONIZED a une
# couverture globale fiable (le catalogue SR ne remonte pas à 2016).
EARLY_START, EARLY_END, EARLY_LABEL = "2017-01-01", "2017-12-31", "2017"

CLOUD_PCT_S2 = 20

BOUNDARY_PATH = "data/raw/app_boundary.geojson"

# Sites culturels : rayon de buffer et seuil de "perte significative"
SITE_BUFFER_METERS = 1000
LOSS_THRESHOLD = -0.1  # dNDVI < -0.1 = perte de végétation jugée significative


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


@st.cache_data(ttl=3600, show_spinner=False)
def sentinel2_ndvi_tile(aoi_geojson_str, start, end, cloud_pct):
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    n = coll.size().getInfo()
    if n == 0:
        return None, 0
    image = coll.median()
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI").clip(aoi)
    tile = ndvi.getMapId({"min": -1, "max": 1, "palette": NDVI_PALETTE})
    return tile["tile_fetcher"].url_format, n


def _s2_ndvi_diff_image(aoi, early_start, early_end, recent_start, recent_end):
    """Image dNDVI (recent - early) Sentinel-2, clippée à aoi. None si pas d'images."""
    early_coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(early_start, early_end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_S2))
    )
    recent_coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(recent_start, recent_end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_S2))
    )
    if early_coll.size().getInfo() == 0 or recent_coll.size().getInfo() == 0:
        return None

    early_ndvi = early_coll.median().normalizedDifference(["B8", "B4"]).rename("NDVI")
    recent_ndvi = recent_coll.median().normalizedDifference(["B8", "B4"]).rename("NDVI")
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
def vegetation_loss_tile(
    aoi_geojson_str, early_start, early_end, recent_start, recent_end, threshold
):
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
# =========================================================
tile_recent = tile_early = tile_diff = None
n_recent = n_early = 0

if active_gdf is not None:
    aoi_ee = geopandas_to_ee(active_gdf)
    aoi_geojson_str = json.dumps(aoi_ee.getInfo())

    try:
        tile_recent, n_recent = sentinel2_ndvi_tile(
            aoi_geojson_str, RECENT_START, RECENT_END, CLOUD_PCT_S2
        )
        tile_early, n_early = sentinel2_ndvi_tile(
            aoi_geojson_str, EARLY_START, EARLY_END, CLOUD_PCT_S2
        )
        tile_diff = ndvi_diff_tile(aoi_geojson_str, EARLY_START, EARLY_END, RECENT_START, RECENT_END)
    except Exception as e:
        st.error(f"Earth Engine computation failed: {e}")

    st.caption(
        f"Sentinel-2 images ({RECENT_LABEL}): {n_recent} | "
        f"Sentinel-2 images ({EARLY_LABEL}): {n_early}"
    )


# =========================================================
# SITES CULTURELS : parsing, buffer 1km, masque de perte NDVI
# =========================================================
sites_gdf = None
buffers_gdf = None
tile_loss = None
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
        tile_loss = vegetation_loss_tile(
            buffer_geojson_str, EARLY_START, EARLY_END, RECENT_START, RECENT_END, LOSS_THRESHOLD
        )
        if tile_loss is None:
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

if tile_recent:
    add_tile_layer(m, tile_recent, f"NDVI {RECENT_LABEL} (Sentinel-2)", show=True)
if tile_early:
    add_tile_layer(m, tile_early, f"NDVI {EARLY_LABEL} (Sentinel-2)", show=False)
if tile_diff:
    add_tile_layer(m, tile_diff, f"Vegetation change {EARLY_LABEL} → {RECENT_LABEL}", show=False)

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

if tile_loss:
    add_tile_layer(m, tile_loss, "Vegetation loss near sites (dNDVI < %.2f)" % LOSS_THRESHOLD, show=True)

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

if tile_recent or tile_early or tile_diff or tile_loss:
    folium.LayerControl(collapsed=False).add_to(m)
else:
    folium.LayerControl().add_to(m)

m.fit_bounds([[miny, minx], [maxy, maxx]])

if tile_recent:
    m.get_root().html.add_child(folium.Element(
        legend_html(f"NDVI {RECENT_LABEL}", "Low", "High", NDVI_PALETTE, bottom=30)
    ))
if tile_diff:
    m.get_root().html.add_child(folium.Element(
        legend_html(f"Change {EARLY_LABEL}→{RECENT_LABEL}", "Loss", "Gain", DIFF_PALETTE, bottom=140)
    ))
if tile_loss:
    loss_legend = f"""
    <div style="position: fixed; bottom: 250px; left: 30px; width: 260px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:11px; padding:7px;">
      <b>Sites — vegetation loss</b><br><br>
      <span style="display:inline-block; width:14px; height:14px;
                   background:#ff0000; margin-right:6px;"></span>
      dNDVI &lt; {LOSS_THRESHOLD} within {SITE_BUFFER_METERS} m buffer
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
