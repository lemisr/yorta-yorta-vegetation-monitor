import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Yorta Yorta Vegetation Monitor",
    page_icon="🌿",
    layout="wide"
)

# Sidebar
st.sidebar.title("🌿 Yorta Yorta Monitor")

st.sidebar.markdown(
    """
    ### Navigation
    
    - Overview
    - Vegetation change
    - Satellite imagery
    - Reports
    """
)

# Titre principal
st.title("🌿 Yorta Yorta Vegetation Monitor")

st.markdown(
    """
    This application explores vegetation dynamics across Yorta Yorta Country
    using GIS and remote sensing techniques.
    
    The objective is to monitor vegetation change through time and provide
    accessible spatial information.
    """
)

# Séparation
st.divider()

# Section projet
st.header("Project overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Study area",
        value="Yorta Yorta Country"
    )

with col2:
    st.metric(
        label="Data",
        value="Satellite imagery"
    )

with col3:
    st.metric(
        label="Method",
        value="Remote sensing"
    )


# Future map section
st.divider()

st.header("Interactive map")

st.info(
    "🗺️ Map will be added here using GeoPandas and Folium."
)


# Future analysis section
st.header("Vegetation analysis")

st.info(
    "📊 NDVI trends and vegetation change statistics will appear here."
)
