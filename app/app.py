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

st.markdown("""
<style>
.stApp{
   background-color: #1e1e1e;
   color: #f0f0f0;
}
section[data-testid="stSidebar"] {
  background-color: #262626;
}
section[data-testid="stSidebar"] * {
  color: #f0f0f0 !important;
}
[data_testid="stFileuploader"] {
  background-color: #2a2a2a;
  border-radius: 8px;
  padding: 10px;
}
[data-testid="stFileUploaderDropzone"] {
  background-color: #2a2a2a !important;
  color: #f0f0f0 !important;
}
[data-testid="stFileUploaderDropzone"] * {
  color: #f0f0f0 !important;
}
.stButton button, .stDownloadButton button, [data-testid="stFileUploaderDropzone"] button {
   background-color: #2a2a2a !important;
   color: #f0f0f0 !important;
   border: 1px solid #444 !important;
}
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
   background-color: #383838 !important;
   border: 1px solid #666 !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p {
  color: #f0f0f0 !important;
}

</style>
""", unsafe_allow_html=True)
   


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

VEG_OPACITY = 0.4  # ~60% de transparence pour les couches NDVI, pour laisser voir le fond de carte

ESRI_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ROADS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"

BOUNDARY_PATH = "data/raw/app_boundary.geojson"

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
    combined_geom = gdf.geometry.unary_union
    geojson = json.loads(gpd.GeoSeries([combined_geom], crs=gdf.crs).to_json())
    return ee.Geometry(geojson["features"][0]["geometry"])


def geom_signature(geom_dict):
    """Signature stable d'une géométrie GeoJSON (dict) pour détecter un changement."""
    return json.dumps(geom_dict, sort_keys=True)


def shapely_geom_to_ee(geom):
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


@st.cache_data(ttl=3600, show_spinner=False)
def forest_stats(aoi_geojson_str, forest_spec, threshold=LOSS_THRESHOLD):
    """Statistiques simples (ha) : surface forêt de référence, surface perdue,
    % perdu. Calculées côté serveur EE via reduceRegion (somme des surfaces pixel)."""
    aoi = ee.Geometry(json.loads(aoi_geojson_str))
    forest_mask = get_forest_mask(aoi, forest_spec)
    if forest_mask is None:
        return None

    diff = _s2_ndvi_diff_image(aoi, EARLY_START, EARLY_END, RECENT_START, RECENT_END)
    if diff is None:
        return None
    loss_mask = diff.lt(threshold).selfMask().updateMask(forest_mask)

    pixel_area = ee.Image.pixelArea()
    forest_area = pixel_area.updateMask(forest_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=20, maxPixels=1e13, bestEffort=True, tileScale=8,
    ).get("area")
    loss_area = pixel_area.updateMask(loss_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=20, maxPixels=1e13, bestEffort=True, tileScale=8,
    ).get("area")

    result = ee.Dictionary({"forest_area": forest_area, "loss_area": loss_area}).getInfo()
    forest_ha = (result.get("forest_area") or 0) / 10000
    loss_ha = (result.get("loss_area") or 0) / 10000
    loss_pct = (loss_ha / forest_ha * 100) if forest_ha > 0 else 0
    return {"forest_ha": forest_ha, "loss_ha": loss_ha, "loss_pct": loss_pct}


def _deg2num(lat, lon, zoom):
    """Coordonnées de tuile (flottantes) pour lat/lon à un niveau de zoom donné
    (projection Web Mercator, standard des tuiles XYZ)."""
    import math
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _choose_zoom(minx, miny, maxx, maxy, target_px, max_zoom=18):
    """Le plus grand niveau de zoom dont l'emprise tient dans target_px."""
    for z in range(max_zoom, 0, -1):
        x0, y0 = _deg2num(maxy, minx, z)
        x1, y1 = _deg2num(miny, maxx, z)
        if abs(x1 - x0) * 256 <= target_px and abs(y1 - y0) * 256 <= target_px:
            return z
    return 1


def _fetch_tile(url_template, z, x, y, session):
    """Récupère une tuile 256x256 ; renvoie une tuile transparente si indisponible
    (tuile hors couverture, 404, etc.) plutôt que de faire planter tout le rendu."""
    from PIL import Image as PILImage
    url = url_template.format(z=z, x=x, y=y)
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return PILImage.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return PILImage.new("RGBA", (256, 256), (0, 0, 0, 0))


def _paste_tile_layer(canvas, url_template, zoom, tx_start, tx_end, ty_start, ty_end, gx0, gy0, session, opacity=1.0):
    """Compose une couche de tuiles XYZ complète sur `canvas`, à l'opacité donnée."""
    from concurrent.futures import ThreadPoolExecutor

    coords = [
        (tx,ty)
        for tx in range(tx_start,tx_end +1)
        for ty in range(ty_start,ty_end +1)
    ]
    
    def _fetch(coord):
        tx,ty = coord
        tile = _fetch_tile(url_template, zoom, tx, ty, session)
        if opacity < 1.0:
            alpha = tile.getchannel("A").point(lambda a: int(a * opacity))
            tile.putalpha(alpha)
        return tx, ty, tile

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_fetch,coords))

    for tx, ty, tile in results:

        px, py = tx * 256 - gx0, ty * 256 - gy0
        canvas.alpha_composite(tile, dest=(px, py))

@st.cache_data(ttl=3600, show_spinner=False)
def build_report_image_png(_active_gdf, ndvi_tile_url, loss_tile_url, _sites_gdf, site_name_col, cache_key, margin_ratio=0.3, target_px=2000):
    """Composite le rapport à partir des MÊMES tuiles que celles affichées sur la
    carte (satellite + routes + labels Esri, NDVI récent, perte) — pas de nouveau
    calcul Earth Engine séparé : c'est littéralement un "screenshot" de ce qui est
    déjà montré à l'écran, centré sur la zone étudiée avec une marge de contexte.
    Tous les points de sites importés sont dessinés dessus, pas seulement ceux
    dans la zone d'analyse."""
    import requests
    from PIL import Image as PILImage, ImageDraw

    minx, miny, maxx, maxy = _active_gdf.total_bounds
    span_x, span_y = maxx - minx, maxy - miny
    margin = max(span_x, span_y) * margin_ratio or 0.01
    minx, maxx = minx - margin, maxx + margin
    miny, maxy = miny - margin, maxy + margin

    zoom = _choose_zoom(minx, miny, maxx, maxy, target_px)
    gx0f, gy0f = _deg2num(maxy, minx, zoom)  # coin haut-gauche (pixels globaux, flottant)
    gx1f, gy1f = _deg2num(miny, maxx, zoom)  # coin bas-droit
    gx0, gy0 = int(gx0f * 256), int(gy0f * 256)
    width_px = max(1, int(gx1f * 256) - gx0)
    height_px = max(1, int(gy1f * 256) - gy0)

    tx_start, ty_start = gx0 // 256, gy0 // 256
    tx_end, ty_end = (gx0 + width_px) // 256, (gy0 + height_px) // 256

    canvas = PILImage.new("RGBA", (width_px, height_px), (0, 0, 0, 255))
    session = requests.Session()

    layers = [(ESRI_SATELLITE_URL, 1.0), (ESRI_ROADS_URL, 1.0), (ESRI_LABELS_URL, 1.0)]
    if ndvi_tile_url:
        layers.append((ndvi_tile_url, 1.0))
    if loss_tile_url:
        layers.append((loss_tile_url, 1.0))

    for url_template, opacity in layers:
        _paste_tile_layer(canvas, url_template, zoom, tx_start, tx_end, ty_start, ty_end, gx0, gy0, session, opacity)

    img = canvas.convert("RGB")
    draw = ImageDraw.Draw(img)

    def _pixel(lon, lat):
        px_f, py_f = _deg2num(lat, lon, zoom)
        return px_f * 256 - gx0, py_f * 256 - gy0

    # Contour de la zone étudiée (même style que sur la carte : blanc, pointillé)
    for geom in _active_gdf.geometry:
        rings = [geom.exterior] + list(geom.interiors) if geom.geom_type == "Polygon" else [
            r for poly in geom.geoms for r in [poly.exterior] + list(poly.interiors)
        ]
        for ring in rings:
            pts = [_pixel(lon, lat) for lon, lat in ring.coords]
            draw.line(pts, fill=(0, 0, 0), width=9)
            draw.line(pts, fill=(255, 255, 255), width=3)                

    if _sites_gdf is not None:
        for _, row in _sites_gdf.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue  # site hors du cadre visible de l'image — ne peut pas être dessiné
            px, py = _pixel(lon, lat)
            r = 5
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 140, 0), outline=(0, 0, 0))
            if site_name_col:
                label = str(row[site_name_col])
                draw.text((px + r + 3, py - r), label, fill=(255, 255, 255))
                draw.text((px + r + 2, py - r - 1), label, fill=(0, 0, 0))  # léger contour lisibilité

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def build_pdf_report(spec, site_count, report_image_png=None):
    """PDF : zone, méthode/seuil de masque forêt, dates comparées, stats, et une
    carte (satellite + NDVI récent + perte, avec les sites) si fournie."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 3 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "Yorta Yorta Vegetation Monitor — Report")
    y -= 1 * cm

    c.setFont("Helvetica", 10)
    method_label = "NDVI annual threshold" if spec[0] == "ndvi" else "Dynamic World tree probability"
    lines = [
        f"Period compared: {EARLY_LABEL} vs {RECENT_LABEL}",
        f"Forest mask method: {method_label} (threshold {spec[1]})",
        f"Loss threshold: dNDVI < {LOSS_THRESHOLD}",
        f"Cultural/heritage sites loaded: {site_count}",
        "",
        
    ]
    for line in lines:
        c.drawString(2 * cm, y, line)
        y -= 0.7 * cm

    if report_image_png:
        y -= 0.5 * cm
        img_reader = ImageReader(io.BytesIO(report_image_png))
        img_w, img_h = img_reader.getSize()
        draw_w = width - 4 * cm
        draw_h = draw_w * img_h / img_w
        if y - draw_h < 2 * cm:  # pas assez de place, nouvelle page
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 3 * cm
        c.drawImage(img_reader, 2 * cm, y - draw_h, width=draw_w, height=draw_h)
        y -= draw_h

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


def add_tile_layer(fmap, url, name, show=False, opacity=1.0, control=True):
    folium.TileLayer(
        tiles=url,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=control,
        show=show,
        opacity=opacity,
    ).add_to(fmap)


def legend_html_v2(items, bottom=30):
    """Légende design (carte arrondie, ombre légère). `items` est une liste de dicts :
    {'type': 'gradient', 'title', 'palette', 'low', 'high'} ou {'type': 'swatch', 'title', 'color'}."""
    blocks = []
    for item in items:
        if item["type"] == "gradient":
            gradient = ", ".join(f"#{c}" for c in item["palette"])
            blocks.append(f"""
              <div style="margin-bottom:10px;">
                <div style="font-weight:600; margin-bottom:4px; color:#f0f0f0;">{item['title']}</div>
                <div style="height:10px; border-radius:5px;
                            background: linear-gradient(to right, {gradient});"></div>
                <div style="display:flex; justify-content:space-between; font-size:10px;
                            color:#bbb; margin-top:2px;">
                  <span>{item['low']}</span><span>{item['high']}</span>
                </div>
              </div>
            """)
        elif item["type"] == "swatch":
            blocks.append(f"""
              <div style="display:flex; align-items:center; margin-bottom:4px;">
                <span style="display:inline-block; width:12px; height:12px; border-radius:3px;
                             background:#{item['color']}; margin-right:8px;"></span>
                <span style="font-size:11px; color:#eee;">{item['title']}</span>
              </div>
            """)
    body = "".join(blocks)
    return f"""
    <div style="position: fixed; bottom: {bottom}px; left: 24px; width: 230px;
                background-color: rgba(30,30,30,0.92); border-radius:10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25); z-index:9999;
                font-family: -apple-system, Helvetica, Arial, sans-serif;
                padding:12px 14px;">
      {body}
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

st.subheader("📍 Places")
uploaded_sites_file = st.file_uploader(
    "Upload point locations (KML, zipped Shapefile, or CSV with lat/lon columns)",
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
    options=["Dynamic World"],
    index=0,  # on teste Dynamic World par défaut
)


st.sidebar.caption(
    "Google Dynamic World: pixel-level tree probability from a trained "
    f"Sentinel-2 classifier."
)
dw_threshold = st.sidebar.slider("Tree probability threshold", 0.0, 1.0, 0.35, 0.05)
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

    #Fusionne tous les morceaux en une seule géométrie
    combined_geom = uploaded_area.geometry.unary_union
    uploaded_area = gpd.GeoDataFrame(geometry=[combined_geom], crs=uploaded_area.crs)


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
# CALCUL NDVI + PERTE DE FORET (sur toute l'AOI, si une AOI active et valide existe)
# Une entrée par seuil forêt actif (None = pas de masque, ou [0.5], [0.6], [0.5, 0.6])
# =========================================================
ndvi_layers = []  # liste de dicts : spec, tile_recent, tile_early
loss_layers = []  # liste de dicts : spec, tile_loss, color
n_recent = n_early = 0
aoi_geojson_str = None

if active_gdf is not None:
    aoi_ee = geopandas_to_ee(active_gdf)
    aoi_geojson_str = json.dumps(aoi_ee.getInfo())
    
    try:
        for i, spec in enumerate(active_forest_specs):
            tile_recent, n_recent = sentinel2_ndvi_tile(
                aoi_geojson_str, RECENT_START, RECENT_END, CLOUD_PCT_S2, forest_spec=spec
            )
            tile_early, n_early = sentinel2_ndvi_tile(
                aoi_geojson_str, EARLY_START, EARLY_END, CLOUD_PCT_S2, forest_spec=spec
            )
            ndvi_layers.append({
                "spec": spec,
                "tile_recent": tile_recent,
                "tile_early": tile_early,
            })

            color = LOSS_COLOR_CYCLE[i % len(LOSS_COLOR_CYCLE)]
            tile_loss = vegetation_loss_tile(
                aoi_geojson_str, EARLY_START, EARLY_END, RECENT_START, RECENT_END,
                LOSS_THRESHOLD, forest_spec=spec, color=color,
            )
            loss_layers.append({"spec": spec, "tile_loss": tile_loss, "color": color})
    except Exception as e:
        st.error(f"Earth Engine computation failed: {e}")

    st.caption(
        f"Sentinel-2 images ({RECENT_LABEL}, mai–sept.): {n_recent} | "
        f"Sentinel-2 images ({EARLY_LABEL}, mai–sept.): {n_early}"
    )


# =========================================================
# SITES CULTURELS : parsing, affichage en points (pas de buffer)
# =========================================================
sites_gdf = None
site_name_col = None

if uploaded_sites_file is not None:
    try:
        raw_gdf = parse_sites_file(
            uploaded_sites_file.getvalue(), uploaded_sites_file.name
        )

        # On ne garde que les points dans la limite de l'application (évite tout crash EE)
        inside_mask = raw_gdf.within(boundary_geom)
        excluded = int((~inside_mask).sum())
        sites_gdf = raw_gdf[inside_mask].reset_index(drop=True)

        if excluded:
            st.warning(f"{excluded} point(s) outside the application boundary were ignored.")

        if sites_gdf.empty:
            st.error("❌ No point is inside the application boundary.")
            sites_gdf = None
        else:
            site_name_col = site_label_column(sites_gdf)
            st.success(f"✅ {len(sites_gdf)} point(s) loaded")

    except Exception as e:
        st.error(f"Could not read the points file: {e}")


# =========================================================
# EXPORT PDF (stats + carte, sur le 1er seuil forêt actif)
# =========================================================
if aoi_geojson_str is not None and active_forest_specs and active_forest_specs[0] is not None:
    st.subheader("📄 Report export")
    if st.button("Generate PDF report"):
        with st.spinner("Rendering map..."):
            try:
                spec = active_forest_specs[0]
                
                ndvi_tile_url = ndvi_layers[0]["tile_recent"] if ndvi_layers else None
                loss_tile_url = loss_layers[0]["tile_loss"] if loss_layers else None
                report_png = build_report_image_png(
                    active_gdf, ndvi_tile_url, loss_tile_url, sites_gdf, site_name_col, aoi_geojson_str
                )
                pdf_bytes = build_pdf_report(
                    spec, len(sites_gdf) if sites_gdf is not None else 0,
                    report_image_png=report_png,
                )
                    # Stocké en session_state : sinon le download_button, rendu dans le
                    # même bloc `if st.button(...)`, disparaît dès le rerun déclenché
                    # par son propre clic — un bug classique Streamlit.
                st.session_state["report_pdf_bytes"] = pdf_bytes
                st.session_state.pop("report_error", None)
            except Exception:
                import traceback
                # Stocké en session_state pour la même raison que le PDF ci-dessus :
                # sinon st.error() disparaît au prochain rerun (ex: interaction carte)
                # avant même d'avoir pu le lire.
                st.session_state["report_error"] = traceback.format_exc()
                st.session_state.pop("report_pdf_bytes", None)

    if st.session_state.get("report_error"):
        st.error("Report generation failed:")
        st.code(st.session_state["report_error"])

    if st.session_state.get("report_pdf_bytes"):
        st.download_button(
            "⬇️ Download PDF",
            data=st.session_state["report_pdf_bytes"],
            file_name="vegetation_report.pdf",
            mime="application/pdf",
        )


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
    tiles=ESRI_SATELLITE_URL,
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False,
).add_to(m)

# Routes + noms de villes en surimpression du satellite (Esri Reference services)
folium.TileLayer(
    tiles=ESRI_ROADS_URL,
    attr="Esri",
    name="Roads",
    overlay=True,
    show=True,
).add_to(m)
folium.TileLayer(
    tiles=ESRI_LABELS_URL,
    attr="Esri",
    name="Place labels",
    overlay=True,
    show=True,
).add_to(m)

folium.GeoJson(
    boundary,
    name="Application boundary",
    style_function=lambda x: {"fillColor": "transparent", "color": "black", "weight": 1},
    control=False,  # démo : toujours visible, pas de case à cocher
).add_to(m)

if active_gdf is not None:
    folium.GeoJson(
        active_gdf[["geometry"]].to_json(),
        name="Selected area",
        style_function=lambda x: {
            "color": "#ffffff", "weight": 2, "fillOpacity": 0,
            "className":"selected-area-outline",
        },
    ).add_to(m)


def _suffix(spec):
    if spec is None:
        return ""
    method, value = spec
    label = "NDVI" if method == "ndvi" else "DW trees"
    return f" — {label}≥{value}"


for i, layer in enumerate(ndvi_layers):
    spec = layer["spec"]
    
    if layer["tile_recent"]:
        add_tile_layer(
            m, layer["tile_recent"], f"Forest {RECENT_LABEL}{_suffix(spec)} (Sentinel-2)",
            show=True, control=True,
        )
    if layer["tile_early"]:
        add_tile_layer(
            m, layer["tile_early"], f"Forest {EARLY_LABEL}{_suffix(spec)} (Sentinel-2)",
            show=True, control=True,
        )

if sites_gdf is not None:
    sites_layer = folium.FeatureGroup(name="Places", show=True)
    for i, row in sites_gdf.iterrows():
        label = str(row[site_name_col]) if site_name_col else f"Point {i + 1}"
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=label,
            tooltip=label,
            icon=folium.DivIcon(html=f'<div style="display:flex;align-items:center;white-space:nowrap;"><span style="background-color:white;border:2px solid black;border-radius:50%;width:14px;height:14px;display:inline-block;"></span><span style="margin-left:4px;color:white;font-weight:bold;text-shadow:1px 1px 2px black;">{label}</span></div>'),
        ).add_to(sites_layer)
    sites_layer.add_to(m)

for layer in loss_layers:
    if layer["tile_loss"]:
        add_tile_layer(
            m, layer["tile_loss"],
            f"Forest loss 2019-2025",
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
    edit_options={
        "edit":False,
        "remove":False,
    },
)
draw.add_to(m)

any_ee_layer = any(l["tile_recent"] or l["tile_early"] for l in ndvi_layers) or any(
    l["tile_loss"] for l in loss_layers
)
if any_ee_layer:
    folium.LayerControl(collapsed=False).add_to(m)
else:
    folium.LayerControl().add_to(m)

layer_control_css = """
<style>
.leaflet-control-layers {
    background-color: rgba(30,30,30,0.92) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    padding: 10px 12px !important;
}
.leaflet-control-layers-list label {
    color: #f0f0f0 !important;
}
.leaflet-draw-toolbar a {
    background-color: #2a2a2a !important;
    border-color: #444 !important;
}
.leaflet-draw-toolbar a:hover {
    background-color: #383838 !important;
}
.leaflet-draw-actions a {
    background-color: #2a2a2a !important;
    color: #f0f0f0 !important;
}
.leaflet-draw-actions a:hover {
    background-color: #383838 !important;
}
.leaflet-control-zoom a {
    background-color: #2a2a2a !important;
    color: #7a7a7a !important;
    border-color: #444 !important;
}
.leaflet-control-zoom a:hover {
    background-color: #383838 !important;
}
.selected-area-outline {
   filter: drop-shadow(0 0 4px rgba(0,0,0,1)) drop-shadow(0 0 4px rgba(0,0,0,1));
}
</style>
"""
m.get_root().html.add_child(folium.Element(layer_control_css))

m.fit_bounds([[miny, minx], [maxy, maxx]])

legend_items = []
if ndvi_layers and ndvi_layers[0]["tile_recent"]:
    legend_items.append({
        "type": "gradient",
        "title": f"Vegetation NDVI",
        "palette": NDVI_PALETTE,
        "low": "Low", "high": "High",
    })
if loss_layers and any(l["tile_loss"] for l in loss_layers):
    for l in loss_layers:
        if l["tile_loss"]:
            legend_items.append({
                "type": "swatch",
                "title": f"Forest loss{_suffix(l['spec'])}",
                "color": l["color"],
            })
if legend_items:
    m.get_root().html.add_child(folium.Element(legend_html_v2(legend_items, bottom=30)))


# =========================================================
# AFFICHAGE (un seul appel st_folium) + capture du dessin
# =========================================================

m.get_root().html.add_child(folium.Element("""
<style>
html, body {
  background-color: #1e1e1e !important;
}
</style>
"""))

col_left, col_center, col_right = st.columns([1, 8, 1])
with col_center:
  
  map_data = st_folium(m, width=1300, height=850, key="main_map")

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
