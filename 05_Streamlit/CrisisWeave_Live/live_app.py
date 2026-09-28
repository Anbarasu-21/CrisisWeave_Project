
# ============================================================
# CRISISWEAVE — LIVE VERIFICATION DASHBOARD
# ============================================================

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
import random


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    "/content/gdrive/MyDrive/"
    "CrisisWeave_Project"
)

LIVE_DIR = os.path.join(
    PROJECT_ROOT,
    "05_Streamlit",
    "CrisisWeave_Live"
)

if LIVE_DIR not in sys.path:
    sys.path.insert(
        0,
        LIVE_DIR
    )


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="CrisisWeave — Live Verification",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# ENGINE
# ============================================================

try:

    import live_engine as cw

except Exception as exc:

    st.error(
        "Live engine could not be imported."
    )

    st.exception(
        exc
    )

    st.stop()


# ============================================================
# CACHED MODEL LOADING
# ============================================================

@st.cache_resource(
    show_spinner="Loading CrisisWeave AI models..."
)
def load_models():

    cnn_model = cw.load_cnn()

    bert_tokenizer, bert_model = (
        cw.load_bert()
    )

    return (
        cnn_model,
        bert_tokenizer,
        bert_model
    )


# ============================================================
# SESSION STATE
# ============================================================

if "incident_lat" not in st.session_state:

    st.session_state[
        "incident_lat"
    ] = 9.929117


if "incident_lon" not in st.session_state:

    st.session_state[
        "incident_lon"
    ] = 76.270436


if "analysis_radius" not in st.session_state:

    st.session_state[
        "analysis_radius"
    ] = 1000


if "observations" not in st.session_state:

    st.session_state[
        "observations"
    ] = []


if "last_analysis" not in st.session_state:

    st.session_state[
        "last_analysis"
    ] = None


if "models_loaded" not in st.session_state:

    st.session_state[
        "models_loaded"
    ] = False



# ============================================================
# INCIDENT LOCATION UI SYNC
# ============================================================

def _sync_incident_from_ui():

    st.session_state[
        "incident_lat"
    ] = float(
        st.session_state[
            "ui_incident_lat"
        ]
    )

    st.session_state[
        "incident_lon"
    ] = float(
        st.session_state[
            "ui_incident_lon"
        ]
    )

    # A manual location change makes the
    # previous reconstruction stale.

    st.session_state[
        "last_analysis"
    ] = None



# ============================================================
# LOCATION WIDGET DEFERRED SYNCHRONIZATION
# ============================================================

if "sync_location_widgets" not in st.session_state:

    st.session_state[
        "sync_location_widgets"
    ] = False


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌊 CrisisWeave — Live Disaster Verification"
)

st.caption(
    "Dynamic disaster-state reconstruction "
    "from asynchronous multi-source evidence"
)

st.divider()


# ============================================================
# INCIDENT CONFIGURATION
# ============================================================

st.subheader(
    "1. Incident Configuration"
)

col1, col2, col3 = st.columns(
    [1, 1, 1]
)

# ------------------------------------------------------------
# CRISISWEAVE MAP CLICK WIDGET SYNC
# ------------------------------------------------------------
#
# Streamlit keyed number_input widgets retain their previous
# values across reruns. A map click changes the canonical
# incident coordinates, therefore the keyed widget values
# must be synchronized BEFORE number_input is instantiated.
# ------------------------------------------------------------

if st.session_state.get(
    "map_click_pending",
    False
):

    st.session_state[
        "ui_incident_lat"
    ] = float(
        st.session_state[
            "incident_lat"
        ]
    )

    st.session_state[
        "ui_incident_lon"
    ] = float(
        st.session_state[
            "incident_lon"
        ]
    )

    st.session_state[
        "map_click_pending"
    ] = False


with col1:

    incident_lat = st.number_input(
        "Incident Latitude",
        value=float(
            st.session_state[
                "incident_lat"
            ]
        ),
        format="%.6f",
        key="ui_incident_lat",
        on_change=_sync_incident_from_ui
    )


with col2:

    incident_lon = st.number_input(
        "Incident Longitude",
        value=float(
            st.session_state[
                "incident_lon"
            ]
        ),
        format="%.6f",
        key="ui_incident_lon",
        on_change=_sync_incident_from_ui
    )


with col3:

    radius_options = [
        500,
        1000,
        2000,
        5000
    ]

    current_radius = (
        st.session_state[
            "analysis_radius"
        ]
    )

    if current_radius not in radius_options:

        current_radius = 1000

    radius_m = st.selectbox(
        "GIS Analysis Radius",
        radius_options,
        index=radius_options.index(
            current_radius
        ),
        format_func=lambda x:
            f"{x / 1000:g} km",
        key="ui_radius"
    )


st.session_state[
    "incident_lat"
] = float(
    incident_lat
)

st.session_state[
    "incident_lon"
] = float(
    incident_lon
)

st.session_state[
    "analysis_radius"
] = int(
    radius_m
)

st.info(
    f"Live analysis area: "
    f"{incident_lat:.6f}, "
    f"{incident_lon:.6f} "
    f"within {radius_m / 1000:g} km"
)


# ============================================================
# INCIDENT MAP
# ============================================================

st.subheader(
    "2. Incident Location"
)

try:

    import folium
    from streamlit_folium import st_folium

    # --------------------------------------------------------
    # CANONICAL INCIDENT LOCATION
    # --------------------------------------------------------

    incident_lat = float(
        st.session_state["incident_lat"]
    )

    incident_lon = float(
        st.session_state["incident_lon"]
    )

    # --------------------------------------------------------
    # BUILD MAP
    # --------------------------------------------------------

    incident_map = folium.Map(
        location=[
            incident_lat,
            incident_lon
        ],
        zoom_start=15,
        control_scale=True
    )

    # --------------------------------------------------------
    # INCIDENT PIN
    # --------------------------------------------------------

    folium.Marker(
        [
            incident_lat,
            incident_lon
        ],
        tooltip="Current Incident",
        popup=(
            f"<b>Incident CW-001</b><br>"
            f"Latitude: {incident_lat:.6f}<br>"
            f"Longitude: {incident_lon:.6f}"
        ),
        icon=folium.Icon(
            color="red",
            icon="warning-sign"
        )
    ).add_to(
        incident_map
    )

    # --------------------------------------------------------
    # ANALYSIS RADIUS
    # --------------------------------------------------------

    folium.Circle(
        [
            incident_lat,
            incident_lon
        ],
        radius=radius_m,
        tooltip=(
            f"Analysis radius: {radius_m} m"
        ),
        fill=False
    ).add_to(
        incident_map
    )

    # --------------------------------------------------------
    # MAP CLICK
    # --------------------------------------------------------

    map_result = st_folium(
        incident_map,
        width=None,
        height=500,
        key="incident_location_map",
        returned_objects=[
            "last_clicked"
        ],
        use_container_width=True
    )

    # --------------------------------------------------------
    # PROCESS CLICK
    # --------------------------------------------------------

    clicked = None

    if isinstance(map_result, dict):
        clicked = map_result.get(
            "last_clicked"
        )

    if isinstance(clicked, dict):

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

            clicked_lat = float(clicked_lat)
            clicked_lon = float(clicked_lon)

            click_signature = (
                round(clicked_lat, 7),
                round(clicked_lon, 7)
            )

            previous_click = (
                st.session_state.get(
                    "last_map_click"
                )
            )

            if previous_click != click_signature:

                # ------------------------------------------------
                # STORE CLICK AS NEW INCIDENT LOCATION
                # ------------------------------------------------

                st.session_state[
                    "last_map_click"
                ] = click_signature

                st.session_state[
                    "incident_lat"
                ] = clicked_lat

                st.session_state[
                    "incident_lon"
                ] = clicked_lon

                # ------------------------------------------------
                # INVALIDATE OLD ANALYSIS
                # ------------------------------------------------

                if "last_analysis" in st.session_state:
                    st.session_state[
                        "last_analysis"
                    ] = None

                # ------------------------------------------------
                # ONE CONTROLLED RERUN
                # ------------------------------------------------

                # ------------------------------------------------
                # CRISISWEAVE MAP CLICK PENDING
                # ------------------------------------------------
                st.session_state[
                    "map_click_pending"
                ] = True

                st.rerun()

except Exception as exc:

    st.warning(
        "Interactive map unavailable."
    )

    st.exception(
        exc
    )


# ============================================================
# ADD OBSERVATION
# ============================================================

st.divider()

st.subheader(
    "3. Add Evidence Observation"
)

st.caption(
    "Add image and/or text observations "
    "one at a time. Every observation is "
    "processed immediately and becomes part "
    "of the live evidence stream."
)


obs_col1, obs_col2 = st.columns(
    [1, 1]
)


with obs_col1:

    uploaded_image = st.file_uploader(
        "Upload Disaster Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        accept_multiple_files=False,
        key="observation_image"
    )


with obs_col2:

    observation_text = st.text_area(
        "Observation Text",
        placeholder=(
            "Example: Flood water is rising "
            "rapidly and residents are requesting "
            "rescue assistance."
        ),
        height=150,
        key="observation_text"
    )


# ============================================================
# OBSERVATION METADATA
# ============================================================

meta_col1, meta_col2, meta_col3 = st.columns(
    [1, 1, 1]
)

with meta_col1:

    observed_date = st.date_input(
        "Observed Date",
        value=datetime.now().date(),
        key="observed_date"
    )


with meta_col2:

    observed_time = st.time_input(
        "Observed Time",
        value=datetime.now().time().replace(
            second=0,
            microsecond=0
        ),
        key="observed_time"
    )


with meta_col3:

    obs_same_location = st.checkbox(
        "Use incident location",
        value=True,
        key="obs_same_location"
    )


if obs_same_location:

    obs_lat = incident_lat
    obs_lon = incident_lon

else:

    loc_col1, loc_col2 = st.columns(
        2
    )

    with loc_col1:

        obs_lat = st.number_input(
            "Observation Latitude",
            value=float(
                incident_lat
            ),
            format="%.6f",
            key="obs_lat"
        )

    with loc_col2:

        obs_lon = st.number_input(
            "Observation Longitude",
            value=float(
                incident_lon
            ),
            format="%.6f",
            key="obs_lon"
        )


# ============================================================
# ADD OBSERVATION BUTTON
# ============================================================

add_observation = st.button(
    "➕ ADD OBSERVATION",
    type="primary",
    use_container_width=True
)


if add_observation:

    if (
        uploaded_image is None
        and
        not observation_text.strip()
    ):

        st.error(
            "Provide an image, text, or both."
        )

    else:

        # ----------------------------------------------------
        # Load models only when actual inference is needed.
        # ----------------------------------------------------

        with st.spinner(
            "Loading CrisisWeave models..."
        ):

            (
                cnn_model,
                bert_tokenizer,
                bert_model
            ) = load_models()


        # ----------------------------------------------------
        # Observation metadata
        # ----------------------------------------------------

        observed_datetime = datetime.combine(
            observed_date,
            observed_time
        ).replace(
            tzinfo=ZoneInfo("Asia/Kolkata")
        )

        observation_id = (
            "OBS_"
            +
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%d%H%M%S"
            )
            +
            "_"
            +
            uuid.uuid4().hex[:6]
        )


        # ----------------------------------------------------
        # IMAGE INFERENCE
        # ----------------------------------------------------

        image_bytes = None
        image_result = None
        image_name = None

        if uploaded_image is not None:

            image_bytes = (
                uploaded_image.getvalue()
            )

            image_name = (
                uploaded_image.name
            )

            with st.spinner(
                "Running CNN severity inference..."
            ):

                image_result = (
                    cw.predict_cnn(
                        cnn_model,
                        image_bytes
                    )
                )


        # ----------------------------------------------------
        # TEXT INFERENCE
        # ----------------------------------------------------

        text_value = (
            observation_text.strip()
        )

        text_result = None

        if text_value:

            with st.spinner(
                "Running BERT humanitarian "
                "text inference..."
            ):

                text_result = (
                    cw.predict_bert(
                        bert_tokenizer,
                        bert_model,
                        text_value
                    )
                )


        # ----------------------------------------------------
        # CREATE RUNTIME OBSERVATIONS
        #
        # IMPORTANT:
        # Use the exact constructors from live_engine.py.
        #
        # This guarantees that the Streamlit evidence schema
        # is identical to the schema accepted by
        # reconstruct_state().
        # ----------------------------------------------------

        created_observations = []


        # ----------------------------------------------------
        # IMAGE OBSERVATION
        # ----------------------------------------------------

        if image_result is not None:

            image_observation = (
                cw.create_image_observation(

                    image_bytes=
                        image_bytes,

                    filename=
                        image_name,

                    observed_at=
                        observed_datetime.isoformat(),

                    received_at=
                        (
                        observed_datetime
                        + timedelta(
                            minutes=random.randint(2, 7)
                        )
                    ).isoformat(
                        timespec="seconds"
                    ),

                    latitude=
                        float(obs_lat),

                    longitude=
                        float(obs_lon),

                    cnn_result=
                        image_result
                )
            )

            # Preserve UI/runtime metadata.


            image_observation[
                "image_name"
            ] = image_name

            created_observations.append(
                image_observation
            )


        # ----------------------------------------------------
        # TEXT OBSERVATION
        # ----------------------------------------------------

        if text_result is not None:

            text_observation = (
                cw.create_text_observation(

                    text=
                        text_value,

                    observed_at=
                        observed_datetime.isoformat(),

                    received_at=
                        (
                        observed_datetime
                        + timedelta(
                            minutes=random.randint(2, 7)
                        )
                    ).isoformat(
                        timespec="seconds"
                    ),

                    latitude=
                        float(obs_lat),

                    longitude=
                        float(obs_lon),

                    bert_result=
                        text_result
                )
            )

            # Preserve UI/runtime metadata.


            created_observations.append(
                text_observation
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not created_observations:

            st.error(
                "No valid runtime observation "
                "was created."
            )

            st.stop()


        # ----------------------------------------------------
        # STORE OBSERVATIONS
        #
        # If both image and text were supplied, they are stored
        # as two modality-specific evidence observations.
        #
        # This is intentional:
        #
        # CNN → severity evidence
        # BERT → humanitarian evidence
        # ----------------------------------------------------

        for runtime_observation in (
            created_observations
        ):

            st.session_state[
                "observations"
            ].append(
                runtime_observation
            )

        # ----------------------------------------------------
        # New evidence invalidates the previous reconstruction.
        # ----------------------------------------------------

        st.session_state[
            "last_analysis"
        ] = None


        # ----------------------------------------------------
        # STORE OBSERVATION
        # ----------------------------------------------------


        # Clear previous reconstructed state because
        # a new observation changed the evidence stream.

        st.session_state[
            "last_analysis"
        ] = None


        # ----------------------------------------------------
        # USER FEEDBACK
        # ----------------------------------------------------

        st.success(
            f"{observation_id} added successfully."
        )

        if image_result is not None:

            st.info(
                "CNN severity: "
                +
                image_result[
                    "severity"
                ]
                +
                " — confidence "
                +
                f"{image_result['confidence'] * 100:.1f}%"
            )

        if text_result is not None:

            st.info(
                "BERT category: "
                +
                text_result[
                    "category"
                ]
                +
                " — confidence "
                +
                f"{text_result['confidence'] * 100:.1f}%"
            )

        st.rerun()


# ============================================================
# EVIDENCE TIMELINE
# ============================================================

st.divider()

st.subheader(
    "4. Live Evidence Timeline"
)

observations = st.session_state[
    "observations"
]


if observations:

    rows = []

    for obs in observations:

        prediction = obs.get(
            "prediction"
        )

        if prediction:

            severity = prediction.get(
                "severity"
            )

            category = prediction.get(
                "category"
            )

            confidence = prediction.get(
                "confidence"
            )

            if severity is not None:

                if confidence is not None:
                    result = (
                        f"CNN: {severity} "
                        f"({float(confidence) * 100:.1f}%)"
                    )

                else:
                    result = (
                        f"CNN: {severity}"
                    )

            elif category is not None:

                if confidence is not None:
                    result = (
                        f"BERT: {category} "
                        f"({float(confidence) * 100:.1f}%)"
                    )

                else:
                    result = (
                        f"BERT: {category}"
                    )

            else:

                result = "Prediction available"

        else:

            result = "No prediction"

        rows.append({

            "Observation":
                obs[
                    "observation_id"
                ],

            "Modality":
                obs[
                    "modality"
                ],

            "Observed At":
                obs[
                    "observed_at"
                ],

            "Received At":
                obs[
                    "received_at"
                ],

            "Prediction":
                result,

            "Latitude":
                round(
                    obs[
                        "latitude"
                    ],
                    6
                ),

            "Longitude":
                round(
                    obs[
                        "longitude"
                    ],
                    6
                )
        })

    timeline_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        timeline_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"{len(observations)} runtime observation(s)"
    )

else:

    st.info(
        "No observations added yet."
    )


# ============================================================
# LIVE RECONSTRUCTION
# ============================================================

st.divider()

st.subheader(
    "5. Live Disaster Reconstruction"
)

st.caption(
    "This executes the complete CrisisWeave "
    "pipeline using ONLY the observations currently "
    "stored in this session and live GIS data."
)


run_analysis = st.button(
    "🚨 RUN LIVE RECONSTRUCTION",
    type="primary",
    use_container_width=True
)


if run_analysis:

    if not observations:

        st.error(
            "Add at least one image or text "
            "observation first."
        )

    else:

        with st.spinner(
            "Running live CrisisWeave reconstruction..."
        ):

            try:

                # --------------------------------------------
                # LIVE GIS
                # --------------------------------------------

                gis_context = (
                    cw.fetch_live_gis(
                        incident_lat,
                        incident_lon,
                        radius_m
                    )
                )


                # --------------------------------------------
                # RECONSTRUCTION
                # --------------------------------------------

                analysis_time = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                results = (
                    cw.reconstruct_state(
                        observations=
                            observations,

                        gis_context=
                            gis_context,

                        analysis_time=
                            analysis_time
                    )
                )


                # --------------------------------------------
                # STORE LIVE RESULT
                # --------------------------------------------

                st.session_state[
                    "last_analysis"
                ] = results

                st.session_state[
                    "last_gis_context"
                ] = gis_context

                st.success(
                    "Live reconstruction completed."
                )

            except Exception as exc:

                st.error(
                    "Live reconstruction failed."
                )

                st.exception(
                    exc
                )


# ============================================================
# LIVE RESULT SUMMARY
# ============================================================

results = st.session_state[
    "last_analysis"
]


if results is not None:

    st.divider()

    st.subheader(
        "6. Current Live State"
    )

    fusion = results.get(
        "fusion",
        {}
    )

    temporal = results.get(
        "temporal",
        {}
    )

    conflict = results.get(
        "conflict",
        {}
    )

    verification = results.get(
        "verification",
        {}
    )

    dynamic = results.get(
        "dynamic_state",
        {}
    )

    recommendation = results.get(
        "recommendation",
        {}
    )


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(
        4
    )


    with metric1:

        st.metric(
            "Fused Severity",
            fusion.get(
                "fused_severity",
                "UNKNOWN"
            )
        )


    with metric2:

        st.metric(
            "Fusion Confidence",
            f"{fusion.get('confidence', 0) * 100:.1f}%"
        )


    with metric3:

        st.metric(
            "Temporal Trend",
            temporal.get(
                "trend",
                "UNKNOWN"
            )
        )


    with metric4:

        st.metric(
            "Verification",
            verification.get(
                "status",
                "UNKNOWN"
            )
        )


    # --------------------------------------------------------
    # DETAILED STATE
    # --------------------------------------------------------

    detail1, detail2 = st.columns(
        2
    )


    with detail1:

        st.markdown(
            "### Temporal State"
        )

        st.write(
            "Observations:",
            temporal.get(
                "observation_count",
                0
            )
        )

        st.write(
            "Initial severity:",
            temporal.get(
                "initial_severity",
                "UNKNOWN"
            )
        )

        st.write(
            "Latest severity:",
            temporal.get(
                "latest_severity",
                "UNKNOWN"
            )
        )

        st.write(
            "Trend:",
            temporal.get(
                "trend",
                "UNKNOWN"
            )
        )

        st.write(
            "Progression:",
            temporal.get(
                "progression_direction",
                "UNKNOWN"
            )
        )


    with detail2:

        st.markdown(
            "### Evidence Conflict"
        )

        st.write(
            "Status:",
            conflict.get(
                "status",
                "UNKNOWN"
            )
        )

        st.write(
            "Overall conflict:",
            conflict.get(
                "overall_conflict",
                "UNKNOWN"
            )
        )

        st.write(
            "Difference:",
            conflict.get(
                "difference",
                "UNKNOWN"
            )
        )

        st.write(
            "Humanitarian evidence:",
            conflict.get(
                "humanitarian_evidence",
                "UNKNOWN"
            )
        )


    # --------------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------------

    st.markdown(
        "### Reliability"
    )

    reliability = results.get(
        "reliability",
        {}
    )

    st.write(
        "Mean reliability:",
        round(
            reliability.get(
                "mean_reliability",
                0
            ),
            4
        )
    )


    # --------------------------------------------------------
    # DYNAMIC STATE
    # --------------------------------------------------------

    st.markdown(
        "### Dynamic Disaster State"
    )

    current_state = dynamic.get(
        "current_state",
        {}
    )

    st.write(
        "Severity:",
        current_state.get(
            "severity",
            "UNKNOWN"
        )
    )

    st.write(
        "Confidence:",
        current_state.get(
            "confidence",
            0
        )
    )

    st.write(
        "Uncertainty:",
        current_state.get(
            "uncertainty",
            "UNKNOWN"
        )
    )

    st.write(
        "Operational risk:",
        dynamic.get(
            "operational_risk",
            "UNKNOWN"
        )
    )


    # --------------------------------------------------------
    # NEXT LOCATION
    # --------------------------------------------------------

    priority = results.get(
        "priority",
        {}
    )

    selected = priority.get(
        "selected_next_priority"
    )

    st.markdown(
        "### Next Location"
    )

    if selected:

        st.success(
            selected[
                "name"
            ]
            +
            " — "
            +
            selected[
                "feature_type"
            ]
        )

        st.write(
            "Distance:",
            round(
                selected[
                    "distance_m"
                ],
                2
            ),
            "m"
        )

        st.write(
            "Priority score:",
            selected[
                "priority_score"
            ]
        )

        if selected.get(
            "water_distance_m"
        ) is not None:

            st.write(
                "Candidate → water:",
                round(
                    selected[
                        "water_distance_m"
                    ],
                    2
                ),
                "m"
            )

    else:

        st.info(
            "No operational next location selected."
        )


    # --------------------------------------------------------
    # VERIFICATION MAP
    # --------------------------------------------------------

    st.markdown(
        "### Verification Map"
    )

    verification_gis = st.session_state.get(
        "last_gis_context",
        {}
    )

    verification_features = verification_gis.get(
        "features",
        []
    )

    selected_verification = priority.get(
        "selected_next_priority"
    )

    verification_map = folium.Map(
        location=[
            incident_lat,
            incident_lon
        ],
        zoom_start=15,
        control_scale=True
    )

    # --------------------------------------------------------
    # INCIDENT
    # --------------------------------------------------------

    folium.Marker(
        [
            incident_lat,
            incident_lon
        ],
        tooltip="Incident Location",
        popup="<b>Current Incident</b>",
        icon=folium.Icon(
            color="red",
            icon="warning-sign"
        )
    ).add_to(
        verification_map
    )

    # --------------------------------------------------------
    # GIS FEATURES
    # --------------------------------------------------------

    feature_icons = {
        "Hospital": ("blue", "plus"),
        "School": ("green", "education"),
        "Bridge": ("orange", "road"),
        "Road": ("gray", "road"),
        "Waterbody": ("cadetblue", "tint")
    }

    for feature in verification_features:

        lat = feature.get("latitude")
        lon = feature.get("longitude")

        if lat is None or lon is None:
            continue

        feature_type = feature.get(
            "feature_type",
            "Other"
        )

        # Roads are intentionally excluded from the
        # verification map because they create excessive
        # visual clutter and are not next-location candidates.
        if feature_type == "Road":
            continue

        name = feature.get(
            "name",
            "Unnamed"
        )

        distance = feature.get(
            "distance_m"
        )

        color, icon = feature_icons.get(
            feature_type,
            ("gray", "info-sign")
        )

        popup_text = (
            f"<b>{name}</b><br>"
            f"Type: {feature_type}<br>"
            f"Distance: {distance} m"
        )

        folium.Marker(
            [
                float(lat),
                float(lon)
            ],
            tooltip=f"{feature_type}: {name}",
            popup=popup_text,
            icon=folium.Icon(
                color=color,
                icon=icon
            )
        ).add_to(
            verification_map
        )

    # --------------------------------------------------------
    # NEXT VERIFICATION LOCATION
    # --------------------------------------------------------

    if selected_verification:

        next_lat = selected_verification.get(
            "latitude"
        )

        next_lon = selected_verification.get(
            "longitude"
        )

        if (
            next_lat is not None
            and next_lon is not None
        ):

            folium.Marker(
                [
                    float(next_lat),
                    float(next_lon)
                ],
                tooltip=(
                    "⭐ NEXT VERIFICATION — "
                    f"{selected_verification.get('name', 'Unnamed')} "
                    f"({selected_verification.get('feature_type', 'Unknown')})"
                ),
                popup=(
                    "<b>⭐ NEXT VERIFICATION</b><br>"
                    f"<b>Location:</b> "
                    f"{selected_verification.get('name', 'Unnamed')}<br>"
                    f"<b>Type:</b> "
                    f"{selected_verification.get('feature_type', 'Unknown')}<br>"
                    f"<b>Distance:</b> "
                    f"{selected_verification.get('distance_m', 0)} m<br>"
                    f"<b>Priority Score:</b> "
                    f"{selected_verification.get('priority_score', 0)}"
                ),
                icon=folium.Icon(
                    color="purple",
                    icon="star"
                )
            ).add_to(
                verification_map
            )

    st_folium(
        verification_map,
        width=None,
        height=600,
        key="verification_map",
        returned_objects=[],
        use_container_width=True
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    st.markdown(
        "### Recommendation"
    )

    decision = recommendation.get(
        "decision",
        "INSUFFICIENT EVIDENCE"
    )

    explanation = recommendation.get(
        "explanation",
        ""
    )

    st.warning(
        decision
    )

    if explanation:

        st.write(
            explanation
        )
        # ----------------------------------------------------
        # RECOMMENDATION EVIDENCE UI
        # ----------------------------------------------------

        evidence = recommendation.get(
            "evidence",
            {}
        )

        if evidence:

            st.markdown("#### Evidence")

            evidence_items = [
                (
                    "Latest severity",
                    evidence.get("latest_severity", "N/A")
                ),
                (
                    "Fused severity",
                    evidence.get("fused_severity", "N/A")
                ),
                (
                    "Fusion confidence",
                    evidence.get("fusion_confidence", "N/A")
                ),
                (
                    "Uncertainty",
                    evidence.get("uncertainty", "N/A")
                ),
                (
                    "Temporal trend",
                    evidence.get("temporal_trend", "N/A")
                ),
                (
                    "Conflict status",
                    evidence.get("conflict_status", "N/A")
                ),
                (
                    "Verification status",
                    evidence.get("verification_status", "N/A")
                ),
                (
                    "Verification score",
                    evidence.get("verification_score", "N/A")
                ),
            ]

            for label, value in evidence_items:

                st.markdown(
                    f"**{label}:** {value}"
                )


# ============================================================
# CLEAR OBSERVATIONS
# ============================================================

st.divider()

if st.button(
    "🗑️ CLEAR LIVE SESSION",
    use_container_width=True
):

    st.session_state[
        "observations"
    ] = []

    st.session_state[
        "last_analysis"
    ] = None

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "CrisisWeave Live — runtime evidence reconstruction. "
    "Saved result files are not used as analysis inputs."
)
