
import os
import json
import math
import warnings

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

import tensorflow as tf
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE
# ============================================================

st.set_page_config(
    page_title="CrisisWeave",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = (
    "/content/gdrive/MyDrive/"
    "CrisisWeave_Project"
)

CNN_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "03_Models/CNN/best_model.keras"
)

CNN_MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "03_Models/CNN/class_mapping.json"
)

BERT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "03_Models/BERT/HumAID_Flood_BERT"
)

TEMPORAL_PATH = os.path.join(
    PROJECT_ROOT,
    "04_Results/Temporal/"
    "CrisisWeave_Temporal_State_CW001.json"
)

CONFLICT_PATH = os.path.join(
    PROJECT_ROOT,
    "04_Results/Conflict/"
    "CrisisWeave_Conflict_Assessment_CW001_EV001.json"
)

RELIABILITY_PATH = os.path.join(
    PROJECT_ROOT,
    "04_Results/Reliability/"
    "CrisisWeave_Reliability_CW001_EV001.json"
)


# ============================================================
# 3. STUDY AREA
# ============================================================

CENTER_LAT = 9.9283
CENTER_LON = 76.2683

LAT_OFFSET = 0.0045
LON_OFFSET = 0.0045

NORTH = CENTER_LAT + LAT_OFFSET
SOUTH = CENTER_LAT - LAT_OFFSET
EAST = CENTER_LON + LON_OFFSET
WEST = CENTER_LON - LON_OFFSET

DEFAULT_LAT = 9.929117
DEFAULT_LON = 76.270436


# ============================================================
# 4. SESSION STATE
# ============================================================

if "incident_lat" not in st.session_state:

    st.session_state[
        "incident_lat"
    ] = DEFAULT_LAT


if "incident_lon" not in st.session_state:

    st.session_state[
        "incident_lon"
    ] = DEFAULT_LON


if "analysis_results" not in st.session_state:

    st.session_state[
        "analysis_results"
    ] = None


# ============================================================
# 5. HELPERS
# ============================================================

def load_json(path):

    if not os.path.exists(path):
        return {}

    try:

        with open(
            path,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(dl / 2) ** 2
    )

    return (
        2
        * R
        * math.asin(
            math.sqrt(a)
        )
    )


def bearing_deg(
    lat1,
    lon1,
    lat2,
    lon2
):

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dl = math.radians(
        lon2 - lon1
    )

    y = (
        math.sin(dl)
        * math.cos(p2)
    )

    x = (
        math.cos(p1)
        * math.sin(p2)
        -
        math.sin(p1)
        * math.cos(p2)
        * math.cos(dl)
    )

    return (
        math.degrees(
            math.atan2(
                y,
                x
            )
        )
        + 360
    ) % 360


def direction_alignment(
    bearing,
    progression
):

    angles = {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315
    }

    if progression not in angles:

        return 0.0

    target = angles[
        progression
    ]

    difference = abs(
        (
            bearing
            - target
            + 180
        ) % 360 - 180
    )

    if difference <= 22.5:
        return 1.0

    if difference <= 45:
        return 0.8

    if difference <= 90:
        return 0.5

    if difference <= 135:
        return 0.2

    return 0.0


# ============================================================
# 6. MODELS
# ============================================================

@st.cache_resource
def load_cnn():

    return tf.keras.models.load_model(
        CNN_MODEL_PATH
    )


@st.cache_resource
def load_bert():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BERT_MODEL_PATH
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BERT_MODEL_PATH
        )
    )

    model.eval()

    return tokenizer, model


# ============================================================
# 7. CNN
# ============================================================

def predict_cnn(
    image
):

    model = load_cnn()

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (224, 224)
    )

    arr = np.array(
        image
    ).astype(
        "float32"
    )

    arr = np.expand_dims(
        arr,
        axis=0
    )

    prediction = model.predict(
        arr,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(
            prediction
        )
    )

    # IMPORTANT:
    # Locked CrisisWeave mapping
    #
    # 0 -> High
    # 1 -> Low
    # 2 -> Moderate

    mapping = {
        0: "High",
        1: "Low",
        2: "Moderate"
    }

    label = mapping[
        class_index
    ]

    probabilities = {}

    for index, value in enumerate(
        prediction
    ):

        probabilities[
            mapping[index]
        ] = float(value)

    return {

        "class": label,

        "class_index": class_index,

        "confidence": float(
            prediction[
                class_index
            ]
        ),

        "probabilities":
            probabilities
    }


# ============================================================
# 8. BERT
# ============================================================

def predict_bert(
    text
):

    tokenizer, model = (
        load_bert()
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = (
        torch.softmax(
            outputs.logits,
            dim=-1
        )[0]
    )

    class_index = int(
        torch.argmax(
            probabilities
        )
    )

    label = model.config.id2label.get(
        class_index,
        str(class_index)
    )

    return {

        "class": label,

        "class_index": class_index,

        "confidence": float(
            probabilities[
                class_index
            ]
        )
    }


# ============================================================
# 9. OSMnx DATA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def load_osm_data():

    import osmnx as ox

    bbox = (
        WEST,
        SOUTH,
        EAST,
        NORTH
    )

    # ----------------------------
    # ROAD NETWORK
    # ----------------------------

    try:

        graph = (
            ox.graph_from_bbox(
                bbox=bbox,
                network_type="drive",
                simplify=True
            )
        )

    except Exception:

        graph = None


    # ----------------------------
    # OSM FEATURES
    # ----------------------------

    tags = {

        "hospital": {
            "amenity": "hospital"
        },

        "school": {
            "amenity": "school"
        },

        "bridge": {
            "man_made": "bridge"
        },

        "water": {
            "natural": "water"
        }
    }

    layers = {}

    for name, tag in tags.items():

        try:

            layers[name] = (
                ox.features_from_bbox(
                    bbox=bbox,
                    tags=tag
                )
            )

        except Exception:

            layers[name] = None

    return graph, layers


# ============================================================
# 10. NEAREST ROAD
# ============================================================

def get_nearest_road(
    lat,
    lon,
    graph
):

    empty = {

        "feature_type": "Road",

        "name": None,

        "latitude": None,

        "longitude": None,

        "distance_m": None
    }

    if graph is None:

        return empty

    try:

        import osmnx as ox

        node = (
            ox.distance.nearest_nodes(
                graph,
                X=lon,
                Y=lat
            )
        )

        node_data = (
            graph.nodes[node]
        )

        road_lat = float(
            node_data["y"]
        )

        road_lon = float(
            node_data["x"]
        )

        distance = haversine_m(
            lat,
            lon,
            road_lat,
            road_lon
        )

        return {

            "feature_type":
                "Road",

            "name":
                "Nearest Road",

            "latitude":
                road_lat,

            "longitude":
                road_lon,

            "distance_m":
                round(
                    distance,
                    2
                )
        }

    except Exception:

        return empty


# ============================================================
# 11. NEAREST OSM FEATURE
# ============================================================

def get_nearest_feature(
    lat,
    lon,
    gdf,
    feature_type
):

    empty = {

        "feature_type":
            feature_type,

        "name":
            None,

        "latitude":
            None,

        "longitude":
            None,

        "distance_m":
            None
    }

    if gdf is None:
        return empty

    if len(gdf) == 0:
        return empty

    try:

        import geopandas as gpd

        work = gdf.copy()

        if work.crs is None:

            work = work.set_crs(
                "EPSG:4326"
            )

        projected = (
            work.to_crs(
                "EPSG:32643"
            )
        )

        point = (
            gpd.GeoSeries(
                gpd.points_from_xy(
                    [lon],
                    [lat]
                ),
                crs="EPSG:4326"
            )
            .to_crs(
                "EPSG:32643"
            )
            .iloc[0]
        )

        distances = (
            projected.geometry
            .distance(
                point
            )
        )

        index = (
            distances.idxmin()
        )

        row = projected.loc[
            index
        ]

        distance = float(
            distances.loc[
                index
            ]
        )

        geometry = row.geometry

        if geometry is None:

            return empty

        centroid = (
            geometry.centroid
        )

        centroid_geo = (
            gpd.GeoSeries(
                [centroid],
                crs="EPSG:32643"
            )
            .to_crs(
                "EPSG:4326"
            )
            .iloc[0]
        )

        name = row.get(
            "name",
            None
        )

        if isinstance(
            name,
            list
        ):

            name = (
                name[0]
                if name
                else None
            )

        if pd.isna(name):

            name = None

        return {

            "feature_type":
                feature_type,

            "name":
                str(name)
                if name
                else feature_type,

            "latitude":
                float(
                    centroid_geo.y
                ),

            "longitude":
                float(
                    centroid_geo.x
                ),

            "distance_m":
                round(
                    distance,
                    2
                )
        }

    except Exception:

        return empty


# ============================================================
# 12. DYNAMIC GIS
# ============================================================

def build_gis_context(
    lat,
    lon
):

    graph, layers = (
        load_osm_data()
    )

    road = get_nearest_road(
        lat,
        lon,
        graph
    )

    hospital = (
        get_nearest_feature(
            lat,
            lon,
            layers.get(
                "hospital"
            ),
            "Hospital"
        )
    )

    school = (
        get_nearest_feature(
            lat,
            lon,
            layers.get(
                "school"
            ),
            "School"
        )
    )

    bridge = (
        get_nearest_feature(
            lat,
            lon,
            layers.get(
                "bridge"
            ),
            "Bridge"
        )
    )

    water = (
        get_nearest_feature(
            lat,
            lon,
            layers.get(
                "water"
            ),
            "Waterbody"
        )
    )

    water_distance = (
        water.get(
            "distance_m"
        )
    )

    if water_distance is None:

        relationship = (
            "Unknown"
        )

    elif water_distance <= 50:

        relationship = (
            "Adjacent to Waterbody"
        )

    elif water_distance <= 200:

        relationship = (
            "Near Waterbody"
        )

    else:

        relationship = (
            "Far from Waterbody"
        )

    water[
        "relationship"
    ] = relationship

    return {

        "incident_location": {

            "latitude": lat,

            "longitude": lon
        },

        "nearest_road": road,

        "nearest_hospital":
            hospital,

        "nearest_school":
            school,

        "nearest_bridge":
            bridge,

        "nearest_waterbody":
            water
    }


# ============================================================
# 13. LOCATION VERIFICATION
# ============================================================

def verify_location(
    gis
):

    road = gis.get(
        "nearest_road",
        {}
    )

    hospital = gis.get(
        "nearest_hospital",
        {}
    )

    school = gis.get(
        "nearest_school",
        {}
    )

    bridge = gis.get(
        "nearest_bridge",
        {}
    )

    water = gis.get(
        "nearest_waterbody",
        {}
    )

    score = 0.0

    road_distance = (
        road.get(
            "distance_m"
        )
    )

    water_distance = (
        water.get(
            "distance_m"
        )
    )

    hospital_distance = (
        hospital.get(
            "distance_m"
        )
    )

    school_distance = (
        school.get(
            "distance_m"
        )
    )

    bridge_distance = (
        bridge.get(
            "distance_m"
        )
    )

    # Road
    if road_distance is not None:

        if road_distance <= 50:

            score += 0.25

        elif road_distance <= 100:

            score += 0.15

    # Water
    if water_distance is not None:

        if water_distance <= 50:

            score += 0.30

        elif water_distance <= 200:

            score += 0.20

    # Hospital
    if (
        hospital_distance is not None
        and hospital_distance <= 500
    ):

        score += 0.15

    # School
    if (
        school_distance is not None
        and school_distance <= 500
    ):

        score += 0.10

    # Bridge
    if (
        bridge_distance is not None
        and bridge_distance <= 500
    ):

        score += 0.10

    available = sum(
        x is not None
        for x in [
            road_distance,
            water_distance,
            hospital_distance,
            school_distance,
            bridge_distance
        ]
    )

    if available >= 4:

        score += 0.10

    score = min(
        score,
        1.0
    )

    if score >= 0.75:

        status = "VERIFIED"

    elif score >= 0.50:

        status = (
            "PARTIALLY VERIFIED"
        )

    else:

        status = "UNVERIFIED"

    return {

        "verification_status":
            status,

        "verification_score":
            round(
                score,
                3
            )
    }


# ============================================================
# 14. PRIORITY
# ============================================================

def proximity_score(
    distance
):

    if distance is None:
        return 0.0

    if distance <= 100:
        return 1.0

    if distance <= 500:
        return 0.5

    if distance <= 1000:
        return 0.2

    return 0.0


def water_proximity_score(
    distance
):

    if distance is None:
        return 0.0

    if distance <= 50:
        return 1.0

    if distance <= 200:
        return 0.5

    return 0.0


def criticality(
    feature_type
):

    if feature_type == "Hospital":
        return 1.00

    if feature_type == "Bridge":
        return 0.90

    if feature_type == "School":
        return 0.75

    return 0.50


def calculate_priority(
    lat,
    lon,
    gis,
    progression
):

    candidates = []

    water_distance = (
        gis.get(
            "nearest_waterbody",
            {}
        )
        .get(
            "distance_m"
        )
    )

    wp = (
        water_proximity_score(
            water_distance
        )
    )

    for key in [
        "nearest_hospital",
        "nearest_bridge",
        "nearest_school"
    ]:

        item = gis.get(
            key,
            {}
        )

        target_lat = item.get(
            "latitude"
        )

        target_lon = item.get(
            "longitude"
        )

        distance = item.get(
            "distance_m"
        )

        if (
            target_lat is None
            or target_lon is None
        ):

            continue

        feature_type = item.get(
            "feature_type",
            "Unknown"
        )

        name = item.get(
            "name",
            feature_type
        )

        bearing = bearing_deg(
            lat,
            lon,
            target_lat,
            target_lon
        )

        alignment = (
            direction_alignment(
                bearing,
                progression
            )
        )

        incident_proximity = (
            proximity_score(
                distance
            )
        )

        base_criticality = (
            criticality(
                feature_type
            )
        )

        score = (
            0.40 * alignment
            +
            0.30 * incident_proximity
            +
            0.20 * wp
            +
            0.10 * base_criticality
        )

        candidates.append({

            "name":
                name,

            "feature_type":
                feature_type,

            "latitude":
                target_lat,

            "longitude":
                target_lon,

            "distance_m":
                round(
                    distance,
                    2
                ),

            "bearing_deg":
                round(
                    bearing,
                    2
                ),

            "direction_alignment":
                alignment,

            "incident_proximity":
                incident_proximity,

            "water_proximity":
                wp,

            "base_criticality":
                base_criticality,

            "priority_score":
                round(
                    score,
                    3
                )
        })

    candidates.sort(
        key=lambda x:
            x[
                "priority_score"
            ],
        reverse=True
    )

    selected = (
        candidates[0]
        if candidates
        else None
    )

    return {

        "progression_direction":
            progression,

        "selected_next_priority":
            selected,

        "ranked_candidates":
            candidates
    }


# ============================================================
# 15. LOAD EXISTING STATES
# ============================================================

temporal = load_json(
    TEMPORAL_PATH
)

conflict = load_json(
    CONFLICT_PATH
)

reliability = load_json(
    RELIABILITY_PATH
)


# ============================================================
# 16. HEADER
# ============================================================

st.title(
    "🌊 CrisisWeave"
)

st.caption(
    "Conflict-Aware Framework for "
    "Dynamic Disaster State Reconstruction"
)

st.markdown(
    """
CNN → BERT → GIS → Temporal → Evidence →
Conflict → Reliability → Fusion →
Dynamic State → Location Verification →
Next Priority Location
"""
)


# ============================================================
# 17. SIDEBAR
# ============================================================

st.sidebar.header(
    "Incident Location"
)

manual_lat = st.sidebar.number_input(
    "Latitude",
    value=float(
        st.session_state[
            "incident_lat"
        ]
    ),
    format="%.6f"
)

manual_lon = st.sidebar.number_input(
    "Longitude",
    value=float(
        st.session_state[
            "incident_lon"
        ]
    ),
    format="%.6f"
)

if st.sidebar.button(
    "Apply Coordinates"
):

    st.session_state[
        "incident_lat"
    ] = manual_lat

    st.session_state[
        "incident_lon"
    ] = manual_lon

    st.session_state[
        "analysis_results"
    ] = None

    st.rerun()


# ============================================================
# 18. INPUT
# ============================================================

st.header(
    "1. Incident Input"
)

uploaded_file = st.file_uploader(
    "Upload flood image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

report_text = st.text_area(
    "Incident Report",
    value=(
        "Flood water has entered the road "
        "and people are requesting urgent "
        "assistance. Rescue teams are needed "
        "in the affected area."
    ),
    height=120
)


# ============================================================
# 19. INCIDENT MAP
# ============================================================

st.subheader(
    "Incident Location"
)

st.info(
    "Click anywhere on the map to select "
    "the incident location."
)

basemap = st.selectbox(
    "Basemap",
    [
        "Esri World Street Map",
        "OpenStreetMap"
    ]
)

lat = st.session_state[
    "incident_lat"
]

lon = st.session_state[
    "incident_lon"
]

if basemap == "OpenStreetMap":

    tiles = "OpenStreetMap"
    attribution = None

else:

    tiles = (
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Street_Map/MapServer/tile/"
        "{z}/{y}/{x}"
    )

    attribution = "Esri"

m = folium.Map(
    location=[
        lat,
        lon
    ],
    zoom_start=17,
    tiles=tiles,
    attr=attribution,
    max_zoom=22
)

folium.Marker(
    [
        lat,
        lon
    ],
    tooltip="Selected Incident",
    popup=(
        f"Latitude: {lat:.6f}<br>"
        f"Longitude: {lon:.6f}"
    ),
    icon=folium.Icon(
        color="red",
        icon="warning-sign"
    )
).add_to(m)

map_result = st_folium(
    m,
    width=1100,
    height=500,
    returned_objects=[
        "last_clicked"
    ]
)

clicked = map_result.get(
    "last_clicked"
)

if clicked:

    clicked_lat = clicked.get(
        "lat"
    )

    clicked_lon = clicked.get(
        "lng"
    )

    if (
        clicked_lat is not None
        and clicked_lon is not None
    ):

        st.session_state[
            "incident_lat"
        ] = float(
            clicked_lat
        )

        st.session_state[
            "incident_lon"
        ] = float(
            clicked_lon
        )

        st.session_state[
            "analysis_results"
        ] = None

        st.rerun()


st.success(
    "Selected incident: "
    f"{st.session_state['incident_lat']:.6f}, "
    f"{st.session_state['incident_lon']:.6f}"
)


# ============================================================
# 20. RUN ANALYSIS
# ============================================================

run = st.button(
    "🚨 RUN CRISISWEAVE ANALYSIS",
    type="primary",
    use_container_width=True
)

if run:

    if uploaded_file is None:

        st.warning(
            "Please upload a flood image."
        )

    else:

        current_lat = (
            st.session_state[
                "incident_lat"
            ]
        )

        current_lon = (
            st.session_state[
                "incident_lon"
            ]
        )

        with st.spinner(
            "Running CrisisWeave analysis..."
        ):

            # CNN
            image = Image.open(
                uploaded_file
            )

            cnn = predict_cnn(
                image
            )

            # BERT
            bert = predict_bert(
                report_text
            )

            # GIS
            try:

                gis = build_gis_context(
                    current_lat,
                    current_lon
                )

                gis_error = None

            except Exception as e:

                gis = {

                    "incident_location": {

                        "latitude":
                            current_lat,

                        "longitude":
                            current_lon
                    },

                    "nearest_road": {},

                    "nearest_hospital": {},

                    "nearest_school": {},

                    "nearest_bridge": {},

                    "nearest_waterbody": {}
                }

                gis_error = str(e)

            # Temporal
            progression = temporal.get(
                "progression_direction",
                "NE"
            )

            trend = temporal.get(
                "severity_trend",
                temporal.get(
                    "overall_temporal_trend",
                    "STABLE"
                )
            )

            latest_temporal = temporal.get(
                "latest_severity",
                "High"
            )

            # Fusion
            fused_class = cnn[
                "class"
            ]

            # If temporal agrees,
            # preserve agreement.
            if latest_temporal in [
                "Low",
                "Moderate",
                "High"
            ]:

                if (
                    latest_temporal
                    == fused_class
                ):

                    fused_class = (
                        latest_temporal
                    )

            # Priority
            priority = (
                calculate_priority(
                    current_lat,
                    current_lon,
                    gis,
                    progression
                )
            )

            # Verification
            verification = (
                verify_location(
                    gis
                )
            )

            # Operational state
            if fused_class == "High":

                operational_risk = (
                    "CRITICAL"
                )

            elif fused_class == "Moderate":

                operational_risk = (
                    "HIGH"
                )

            else:

                operational_risk = (
                    "MONITOR"
                )

            # Recommendation
            if (
                fused_class == "High"
                and verification[
                    "verification_status"
                ]
                in [
                    "VERIFIED",
                    "PARTIALLY VERIFIED"
                ]
                and str(
                    trend
                ).upper()
                == "INCREASING"
            ):

                decision = (
                    "IMMEDIATE RESPONSE"
                )

            elif fused_class == "High":

                decision = (
                    "HIGH PRIORITY RESPONSE"
                )

            elif fused_class == "Moderate":

                decision = (
                    "MONITOR AND ASSESS"
                )

            else:

                decision = (
                    "CONTINUE MONITORING"
                )

            st.session_state[
                "analysis_results"
            ] = {

                "incident": {

                    "latitude":
                        current_lat,

                    "longitude":
                        current_lon
                },

                "cnn":
                    cnn,

                "bert":
                    bert,

                "gis":
                    gis,

                "gis_error":
                    gis_error,

                "verification":
                    verification,

                "temporal": {

                    "trend":
                        trend,

                    "progression":
                        progression,

                    "latest_severity":
                        latest_temporal
                },

                "conflict":
                    conflict,

                "reliability":
                    reliability,

                "fusion": {

                    "fused_class":
                        fused_class
                },

                "dynamic_state": {

                    "current_state":
                        fused_class,

                    "operational_risk":
                        operational_risk
                },

                "priority":
                    priority,

                "recommendation": {

                    "decision":
                        decision
                }
            }

        st.success(
            "CrisisWeave analysis completed."
        )


# ============================================================
# 21. RESULTS
# ============================================================

results = st.session_state[
    "analysis_results"
]

if results is not None:

    st.header(
        "2. CrisisWeave Results"
    )

    # ----------------------------
    # LOCATION
    # ----------------------------

    incident = results[
        "incident"
    ]

    st.subheader(
        "📍 Incident Location"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Latitude",
        f"{incident['latitude']:.6f}"
    )

    c2.metric(
        "Longitude",
        f"{incident['longitude']:.6f}"
    )

    # ----------------------------
    # CNN
    # ----------------------------

    st.subheader(
        "🖼️ CNN Flood Severity"
    )

    cnn = results[
        "cnn"
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Severity",
        cnn["class"]
    )

    c2.metric(
        "Confidence",
        f"{cnn['confidence'] * 100:.1f}%"
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Severity": name,
                "Probability (%)":
                    round(
                        value * 100,
                        2
                    )
            }

            for name, value
            in cnn[
                "probabilities"
            ].items()
        ]),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------
    # BERT
    # ----------------------------

    st.subheader(
        "📝 BERT Humanitarian Evidence"
    )

    bert = results[
        "bert"
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Category",
        bert["class"]
    )

    c2.metric(
        "Confidence",
        f"{bert['confidence'] * 100:.1f}%"
    )

    # ----------------------------
    # GIS
    # ----------------------------

    st.subheader(
        "🗺️ GIS Context"
    )

    if results[
        "gis_error"
    ]:

        st.error(
            "GIS error: "
            + str(
                results[
                    "gis_error"
                ]
            )
        )

    gis = results[
        "gis"
    ]

    rows = []

    for key in [
        "nearest_road",
        "nearest_hospital",
        "nearest_school",
        "nearest_bridge",
        "nearest_waterbody"
    ]:

        item = gis.get(
            key,
            {}
        )

        rows.append({

            "Feature":
                item.get(
                    "feature_type",
                    key
                ),

            "Name":
                item.get(
                    "name",
                    "N/A"
                ),

            "Distance (m)":
                item.get(
                    "distance_m",
                    "N/A"
                ),

            "Latitude":
                item.get(
                    "latitude",
                    "N/A"
                ),

            "Longitude":
                item.get(
                    "longitude",
                    "N/A"
                ),

            "Relationship":
                item.get(
                    "relationship",
                    ""
                )
        })

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------
    # VERIFICATION
    # ----------------------------

    st.subheader(
        "📍 Location Verification"
    )

    verification = results[
        "verification"
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Status",
        verification[
            "verification_status"
        ]
    )

    c2.metric(
        "Score",
        f"{verification['verification_score']:.3f}"
    )

    # ----------------------------
    # TEMPORAL
    # ----------------------------

    st.subheader(
        "⏱️ Temporal State"
    )

    temp = results[
        "temporal"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Trend",
        temp["trend"]
    )

    c2.metric(
        "Progression",
        temp["progression"]
    )

    c3.metric(
        "Latest Severity",
        temp["latest_severity"]
    )

    # ----------------------------
    # CONFLICT
    # ----------------------------

    st.subheader(
        "⚔️ Conflict Assessment"
    )

    conflict_result = results[
        "conflict"
    ]

    conflict_status = (
        conflict_result.get(
            "status",
            conflict_result.get(
                "conflict_status",
                "UNKNOWN"
            )
        )
    )

    st.metric(
        "Conflict Status",
        conflict_status
    )

    # ----------------------------
    # FUSION
    # ----------------------------

    st.subheader(
        "🔀 Conflict-Aware Evidence Fusion"
    )

    st.metric(
        "Fused Severity",
        results[
            "fusion"
        ][
            "fused_class"
        ]
    )

    # ----------------------------
    # DYNAMIC STATE
    # ----------------------------

    st.subheader(
        "🚨 Dynamic Disaster State"
    )

    dynamic = results[
        "dynamic_state"
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Current State",
        dynamic[
            "current_state"
        ]
    )

    c2.metric(
        "Operational Risk",
        dynamic[
            "operational_risk"
        ]
    )

    # ----------------------------
    # PRIORITY
    # ----------------------------

    st.subheader(
        "🎯 Next Priority Location"
    )

    priority = results[
        "priority"
    ]

    selected = priority[
        "selected_next_priority"
    ]

    if selected:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Location",
            selected[
                "name"
            ]
        )

        c2.metric(
            "Priority Score",
            f"{selected['priority_score']:.3f}"
        )

        c3.metric(
            "Distance",
            f"{selected['distance_m']:.1f} m"
        )

        st.dataframe(
            pd.DataFrame(
                priority[
                    "ranked_candidates"
                ]
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No suitable priority "
            "location was found."
        )

    # ----------------------------
    # PRIORITY MAP
    # ----------------------------

    st.subheader(
        "🗺️ Next Priority Location Map"
    )

    priority_map = folium.Map(
        location=[
            incident[
                "latitude"
            ],
            incident[
                "longitude"
            ]
        ],
        zoom_start=16,
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr="Esri",
        max_zoom=22
    )

    folium.Marker(
        [
            incident[
                "latitude"
            ],
            incident[
                "longitude"
            ]
        ],
        tooltip="Current Incident",
        icon=folium.Icon(
            color="red",
            icon="warning-sign"
        )
    ).add_to(
        priority_map
    )

    for candidate in priority[
        "ranked_candidates"
    ]:

        folium.CircleMarker(
            location=[
                candidate[
                    "latitude"
                ],
                candidate[
                    "longitude"
                ]
            ],
            radius=7,
            tooltip=(
                f"{candidate['name']} "
                f"— "
                f"{candidate['priority_score']:.3f}"
            ),
            popup=(
                f"<b>{candidate['name']}</b><br>"
                f"Type: "
                f"{candidate['feature_type']}<br>"
                f"Distance: "
                f"{candidate['distance_m']:.1f} m<br>"
                f"Priority: "
                f"{candidate['priority_score']:.3f}"
            ),
            fill=True
        ).add_to(
            priority_map
        )

    st_folium(
        priority_map,
        width=1100,
        height=500
    )

    # ----------------------------
    # RECOMMENDATION
    # ----------------------------

    st.subheader(
        "🧠 CrisisWeave Recommendation"
    )

    decision = results[
        "recommendation"
    ][
        "decision"
    ]

    if decision == (
        "IMMEDIATE RESPONSE"
    ):

        st.error(
            "🚨 IMMEDIATE RESPONSE — "
            "ACT NOW AND PRIORITIZE NEXT LOCATION"
        )

    elif decision == (
        "HIGH PRIORITY RESPONSE"
    ):

        st.warning(
            "⚠️ HIGH PRIORITY RESPONSE"
        )

    else:

        st.info(
            decision
        )

else:

    st.info(
        "Select an incident location, "
        "upload an image and click "
        "RUN CRISISWEAVE ANALYSIS."
    )


# ============================================================
# CRISISWEAVE FINAL INTEGRATION HELPERS
# ============================================================

def cw_safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def cw_distance_to_score(distance_m):
    """
    Convert distance to a bounded proximity score.
    """
    d = cw_safe_float(distance_m)

    if d is None:
        return 0.0

    if d <= 50:
        return 1.0
    elif d <= 100:
        return 0.8
    elif d <= 250:
        return 0.6
    elif d <= 500:
        return 0.4
    elif d <= 1000:
        return 0.2
    else:
        return 0.0


def cw_verification_score(gis_context):
    """
    Transparent spatial verification score.

    Evidence:
      Road      <=100m  -> +0.15
      Water     <=200m  -> +0.25
      Hospital  <=500m  -> +0.20
      School    <=500m  -> +0.15
      Bridge    <=500m  -> +0.15
      Multiple spatial evidence -> +0.10

    Maximum = 1.00
    """

    if not isinstance(gis_context, dict):
        return 0.0

    score = 0.0
    evidence_count = 0

    road = gis_context.get("nearest_road", {}) or {}
    hospital = gis_context.get("nearest_hospital", {}) or {}
    school = gis_context.get("nearest_school", {}) or {}
    bridge = gis_context.get("nearest_bridge", {}) or {}
    water = gis_context.get("nearest_waterbody", {}) or {}

    road_d = cw_safe_float(road.get("distance_m"))
    hospital_d = cw_safe_float(hospital.get("distance_m"))
    school_d = cw_safe_float(school.get("distance_m"))
    bridge_d = cw_safe_float(bridge.get("distance_m"))
    water_d = cw_safe_float(water.get("distance_m"))

    if road_d is not None and road_d <= 100:
        score += 0.15
        evidence_count += 1

    if water_d is not None and water_d <= 200:
        score += 0.25
        evidence_count += 1

    if hospital_d is not None and hospital_d <= 500:
        score += 0.20
        evidence_count += 1

    if school_d is not None and school_d <= 500:
        score += 0.15
        evidence_count += 1

    if bridge_d is not None and bridge_d <= 500:
        score += 0.15
        evidence_count += 1

    if evidence_count >= 3:
        score += 0.10

    return round(min(score, 1.0), 3)


def cw_verification_status(score):
    if score >= 0.75:
        return "VERIFIED"
    elif score >= 0.50:
        return "PARTIALLY VERIFIED"
    else:
        return "UNVERIFIED"


def cw_extract_latlon(item):
    """
    Safely extract a real geographic coordinate pair.
    """
    if not isinstance(item, dict):
        return None, None

    lat = cw_safe_float(
        item.get("latitude", item.get("lat"))
    )

    lon = cw_safe_float(
        item.get("longitude", item.get("lon", item.get("lng")))
    )

    if lat is None or lon is None:
        return None, None

    # Valid WGS84 geographic range
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None

    return lat, lon


def cw_fix_waterbody_display(waterbody, incident_lat, incident_lon):
    """
    Correct waterbody display coordinates.

    The actual distance calculation should remain untouched.
    If a representative coordinate is inconsistent with the
    incident distance, do not expose a misleading coordinate.
    """

    if not isinstance(waterbody, dict):
        return waterbody

    result = dict(waterbody)

    lat, lon = cw_extract_latlon(result)

    distance_m = cw_safe_float(result.get("distance_m"))

    # If coordinate is clearly suspicious, hide the misleading
    # coordinate while retaining the actual calculated distance.
    if lat is not None and lon is not None and distance_m is not None:

        # Approximate straight-line sanity check
        try:
            dlat = (lat - incident_lat) * 111320.0
            dlon = (
                (lon - incident_lon)
                * 111320.0
                * math.cos(math.radians(incident_lat))
            )

            coordinate_distance = math.sqrt(
                dlat ** 2 + dlon ** 2
            )

            # If coordinate and stored distance disagree by
            # more than a large tolerance, suppress coordinates.
            if coordinate_distance > max(distance_m * 5, distance_m + 500):
                result["latitude"] = None
                result["longitude"] = None
                result["coordinate_note"] = (
                    "Representative coordinate omitted; "
                    "nearest-geometry distance retained."
                )

        except Exception:
            pass

    return result


def cw_candidate_reason(candidate):
    reasons = []

    distance = cw_safe_float(candidate.get("distance_m"))
    water_distance = cw_safe_float(candidate.get("water_distance_m"))
    criticality = cw_safe_float(candidate.get("base_criticality"))

    if distance is not None and distance <= 1000:
        reasons.append("Near incident")

    if water_distance is not None and water_distance <= 200:
        reasons.append("Near water")

    if criticality is not None and criticality >= 0.90:
        reasons.append("Critical facility")

    if not reasons:
        reasons.append("Spatial priority candidate")

    return reasons


def cw_can_recommend_priority(verification_status, fused_severity):
    return (
        str(verification_status).upper()
        in {"VERIFIED", "PARTIALLY VERIFIED"}
        and str(fused_severity).upper() == "HIGH"
    )


def cw_conflict_display(conflict_state):
    """
    Preserve UNKNOWN when no dynamic conflict assessment exists.
    Never fabricate AGREEMENT.
    """

    if not isinstance(conflict_state, dict):
        return {
            "status": "UNKNOWN",
            "overall_conflict": None,
            "temporal_tension": "UNKNOWN",
            "humanitarian_evidence": "UNKNOWN",
        }

    status = conflict_state.get(
        "status",
        conflict_state.get("conflict_status", "UNKNOWN")
    )

    overall = conflict_state.get(
        "overall_conflict",
        conflict_state.get("conflict", None)
    )

    temporal = conflict_state.get(
        "temporal_tension",
        "UNKNOWN"
    )

    humanitarian = conflict_state.get(
        "humanitarian_evidence",
        "UNKNOWN"
    )

    return {
        "status": str(status),
        "overall_conflict": overall,
        "temporal_tension": str(temporal),
        "humanitarian_evidence": str(humanitarian),
    }
