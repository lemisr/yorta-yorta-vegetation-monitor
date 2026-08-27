// 1. FOND DE CARTE HYBRIDE (Satellite + Routes + Villes)
Map.setOptions('HYBRID');

// Définissez votre zone d'étude
// var roi = geometry; 
Map.centerObject(roi, 10);

// Périodes
var debut_P1 = '2019-01-01';
var fin_P1   = '2020-12-31';
var debut_P2 = '2024-01-01';
var fin_P2   = '2025-12-31';

// Fonction Cloud Mask Sentinel-2
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000)
              .copyProperties(image, ["system:time_start"]);
}

// Fonction pour calculer le NDVI sur une image unique
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

// 2. CHARGEMENT DES COLLECTIONS COMPLÈTES (Pour analyser la variabilité)
var collection_P1 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(debut_P1, fin_P1)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds)
  .map(addNDVI);

var collection_P2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(debut_P2, fin_P2)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds)
  .map(addNDVI);

// 3. CRÉATION DU MASQUE FORESTIER MAISON (Filtre anti-cultures)
// Les cultures varient beaucoup en 2 ans (Ecart-type élevé). La forêt reste stable.
var ndvi_std_P1 = collection_P1.select('NDVI').reduce(ee.Reducer.stdDev());
var ndvi_median_P1 = collection_P1.select('NDVI').median();

// On garde les pixels où le NDVI est stable (stdDev < 0.08) ET haut (median > 0.6)
var masqueForetSentinel = ndvi_std_P1.lt(0.08).and(ndvi_median_P1.gt(0.6)).clip(roi);

// 4. GÉNÉRATION DES COMPOSITES FINAUX
var s2_P1_median = collection_P1.median().clip(roi);
var s2_P2_median = collection_P2.median().clip(roi);

var ndvi_P1_foret = s2_P1_median.select('NDVI').updateMask(masqueForetSentinel);
var ndvi_P2_foret = s2_P2_median.select('NDVI').updateMask(masqueForetSentinel);

// 5. CALCUL DU CHANGEMENT REEL EN FORÊT
var changementNDVI = ndvi_P2_foret.subtract(ndvi_P1_foret).rename('Changement');

// 6. AFFICHAGE DE L'APPLICATION
var visNdvi = {min: 0.4, max: 0.8, palette: ['#ece7f2', '#a6bddb', '#016450']};
var visChangement = {min: -0.15, max: 0.15, palette: ['#d73027', '#ffffff', '#1a9850']};

Map.addLayer(s2_P2_median, {bands: ['B4', 'B3', 'B2'], max: 0.3}, '1. Fond de carte Sentinel-2 2024-2025', false);
Map.addLayer(ndvi_P1_foret, visNdvi, '2. NDVI Forêt Stable (2019-2020)', false);
Map.addLayer(ndvi_P2_foret, visNdvi, '3. NDVI Forêt Évolué (2024-2025)', false);
Map.addLayer(changementNDVI, visChangement, '4. Déforestation/Gain réel (Hors cultures)');
