
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
# CLEAR LIVE OBSERVATIONS
# ============================================================

st.divider()

clear_observations = st.button(
    "🗑️ CLEAR LIVE OBSERVATIONS",
    use_container_width=True
)

if clear_observations:

    st.session_state[
        "observations"
    ] = []

    st.session_state[
        "last_analysis"
    ] = None

    st.session_state[
        "last_gis_context"
    ] = {}

    st.success(
        "All live observations and reconstruction results "
        "have been cleared."
    )

    st.rerun()


# ============================================================
# LIVE RECONSTRUCTION
# ============================================================

st.divider()

st.subheader(
    "5. Live Disaster Reconstruction"
)

st.caption(
    "This executes the complete CrisisWeave pipeline "
    "independently for every detected geographic incident."
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
            "Grouping incidents and running "
            "CrisisWeave reconstruction..."
        ):

            try:

                # ------------------------------------------------
                # GROUP OBSERVATIONS INTO INCIDENTS
                # ------------------------------------------------

                incident_groups = (
                    cw.group_observations_by_incident(
                        observations,
                        incident_radius_m=100.0
                    )
                )

                if not incident_groups:

                    raise ValueError(
                        "No valid geographic incident groups "
                        "could be created from the observations."
                    )

                analysis_time = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                incident_results = {}

                # ------------------------------------------------
                # RECONSTRUCT EACH INCIDENT INDEPENDENTLY
                # ------------------------------------------------

                for group in incident_groups:

                    incident_id = group[
                        "incident_id"
                    ]

                    group_lat = float(
                        group["center_lat"]
                    )

                    group_lon = float(
                        group["center_lon"]
                    )

                    group_observations = group[
                        "observations"
                    ]

                    # --------------------------------------------
                    # LIVE GIS FOR THIS INCIDENT
                    # --------------------------------------------

                    group_gis_context = (
                        cw.fetch_live_gis(
                            group_lat,
                            group_lon,
                            radius_m
                        )
                    )

                    # --------------------------------------------
                    # RECONSTRUCTION FOR THIS INCIDENT
                    # --------------------------------------------

                    group_result = (
                        cw.reconstruct_state(
                            observations=
                                group_observations,

                            gis_context=
                                group_gis_context,

                            analysis_time=
                                analysis_time
                        )
                    )

                    # --------------------------------------------
                    # STORE INCIDENT RESULT
                    # --------------------------------------------

                    incident_results[
                        incident_id
                    ] = {
                        "incident_id":
                            incident_id,

                        "center_lat":
                            group_lat,

                        "center_lon":
                            group_lon,

                        "observation_count":
                            group[
                                "observation_count"
                            ],

                        "max_observation_distance_m":
                            group[
                                "max_observation_distance_m"
                            ],

                        "observation_ids": [
                            obs.get(
                                "observation_id"
                            )
                            for obs in group_observations
                        ],

                        "observations":
                            group_observations,

                        "gis_context":
                            group_gis_context,

                        "result":
                            group_result
                    }

                # ------------------------------------------------
                # STORE MULTI-INCIDENT RESULT
                # ------------------------------------------------

                results = {
                    "analysis_time":
                        analysis_time,

                    "incident_count":
                        len(incident_results),

                    "incidents":
                        incident_results
                }

                st.session_state[
                    "last_analysis"
                ] = results

                st.session_state[
                    "last_gis_context"
                ] = {
                    incident_id:
                        data["gis_context"]
                    for incident_id, data
                    in incident_results.items()
                }

                st.success(
                    f"Live reconstruction completed for "
                    f"{len(incident_results)} incident(s)."
                )

            except Exception as exc:

                st.error(
                    "Live reconstruction failed."
                )

                st.exception(exc)



# ============================================================

# ============================================================

# ============================================================
# LIVE RESULT SUMMARY
# ============================================================

results = st.session_state.get(
    "last_analysis"
)

if results is not None:

    st.divider()

    st.subheader(
        "6. Current Live State"
    )

    incident_count = int(
        results.get(
            "incident_count",
            0
        )
    )

    incidents = results.get(
        "incidents",
        {}
    )

    if not isinstance(
        incidents,
        dict
    ):
        incidents = {}

    st.info(
        f"Live reconstruction contains "
        f"{incident_count} incident(s)."
    )

    # ========================================================
    # INCIDENT SELECTOR
    # ========================================================
    #
    # Only one incident is displayed at a time.
    # This prevents Incident 1, Incident 2, Incident 3
    # from becoming one huge page.
    # ========================================================

    incident_ids = list(
        incidents.keys()
    )

    if incident_ids:

        if (
            "selected_incident_id"
            not in st.session_state
            or
            st.session_state[
                "selected_incident_id"
            ] not in incident_ids
        ):

            st.session_state[
                "selected_incident_id"
            ] = incident_ids[0]

        st.markdown(
            "### Select Incident"
        )

        selected_incident_id = st.radio(
            "Incident",
            incident_ids,
            index=incident_ids.index(
                st.session_state[
                    "selected_incident_id"
                ]
            ),
            horizontal=True,
            key="incident_selector"
        )

        st.session_state[
            "selected_incident_id"
        ] = selected_incident_id

        # ----------------------------------------------------
        # SELECTED INCIDENT
        # ----------------------------------------------------

        incident_id = (
            selected_incident_id
        )

        incident_data = incidents.get(
            incident_id,
            {}
        )

        st.divider()

        st.markdown(
            f"## 🚨 {incident_id}"
        )

        # ====================================================
        # INCIDENT METADATA
        # ====================================================

        center_lat = incident_data.get(
            "center_lat"
        )

        center_lon = incident_data.get(
            "center_lon"
        )

        observation_count = incident_data.get(
            "observation_count",
            0
        )

        max_distance = incident_data.get(
            "max_observation_distance_m",
            0.0
        )

        metric1, metric2, metric3 = st.columns(
            3
        )

        with metric1:

            st.metric(
                "Observations",
                observation_count
            )

        with metric2:

            st.metric(
                "Center Latitude",
                (
                    f"{float(center_lat):.6f}"
                    if center_lat is not None
                    else "UNKNOWN"
                )
            )

        with metric3:

            st.metric(
                "Cluster Radius",
                (
                    f"{float(max_distance):.1f} m"
                    if max_distance is not None
                    else "UNKNOWN"
                )
            )

        if (
            center_lat is not None
            and center_lon is not None
        ):

            st.caption(
                f"Incident center: "
                f"{float(center_lat):.6f}, "
                f"{float(center_lon):.6f}"
            )

        # ====================================================
        # OBSERVATION IDS
        # ====================================================

        observation_ids = incident_data.get(
            "observation_ids",
            []
        )

        if observation_ids:

            st.write(
                "Observation IDs:",
                ", ".join(
                    str(x)
                    for x in observation_ids
                    if x is not None
                )
            )

        # ====================================================
        # RECONSTRUCTION RESULT
        # ====================================================

        incident_result = incident_data.get(
            "result",
            {}
        )

        if not isinstance(
            incident_result,
            dict
        ):
            incident_result = {}

        fusion = incident_result.get(
            "fusion",
            {}
        )

        temporal = incident_result.get(
            "temporal",
            {}
        )

        conflict = incident_result.get(
            "conflict",
            {}
        )

        verification = incident_result.get(
            "verification",
            {}
        )

        dynamic = incident_result.get(
            "dynamic_state",
            {}
        )

        recommendation = incident_result.get(
            "recommendation",
            {}
        )

        priority = incident_result.get(
            "priority",
            {}
        )

        if not isinstance(
            priority,
            dict
        ):
            priority = {}

        # ====================================================
        # RECONSTRUCTION SUMMARY
        # ====================================================

        st.markdown(
            "### Reconstruction Summary"
        )

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

            try:

                confidence_value = float(
                    fusion.get(
                        "confidence",
                        0
                    )
                )

                confidence_text = (
                    f"{confidence_value * 100:.1f}%"
                )

            except Exception:

                confidence_text = "UNKNOWN"

            st.metric(
                "Fusion Confidence",
                confidence_text
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

        # ====================================================
        # TEMPORAL + CONFLICT
        # ====================================================

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
                    observation_count
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

        # ====================================================
        # RELIABILITY
        # ====================================================

        reliability = incident_result.get(
            "reliability",
            {}
        )

        st.markdown(
            "### Reliability"
        )

        try:

            mean_reliability = float(
                reliability.get(
                    "mean_reliability",
                    0
                )
            )

        except Exception:

            mean_reliability = 0.0

        st.write(
            "Mean reliability:",
            round(
                mean_reliability,
                4
            )
        )

        # ====================================================
        # DYNAMIC DISASTER STATE
        # ====================================================

        st.markdown(
            "### Dynamic Disaster State"
        )

        current_state = dynamic.get(
            "current_state",
            {}
        )

        if not isinstance(
            current_state,
            dict
        ):
            current_state = {}

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

        # ====================================================
        # NEXT VERIFICATION LOCATION
        # ====================================================

        selected_next = priority.get(
            "selected_next_priority"
        )

        st.markdown(
            "### 🎯 Next Verification Location"
        )

        if selected_next:

            next_name = selected_next.get(
                "name",
                "Unnamed location"
            )

            next_type = selected_next.get(
                "feature_type",
                "Unknown"
            )

            next_distance = selected_next.get(
                "distance_m"
            )

            next_score = selected_next.get(
                "priority_score"
            )

            direction_alignment = selected_next.get(
                "direction_alignment"
            )

            st.success(
                f"{next_name} — {next_type}"
            )

            m1, m2, m3 = st.columns(
                3
            )

            with m1:

                if next_distance is not None:

                    st.metric(
                        "Distance",
                        f"{float(next_distance):.1f} m"
                    )

                else:

                    st.metric(
                        "Distance",
                        "UNKNOWN"
                    )

            with m2:

                if next_score is not None:

                    st.metric(
                        "Priority Score",
                        f"{float(next_score):.3f}"
                    )

                else:

                    st.metric(
                        "Priority Score",
                        "UNKNOWN"
                    )

            with m3:

                if direction_alignment is not None:

                    st.metric(
                        "Direction Alignment",
                        f"{float(direction_alignment):.2f}"
                    )

                else:

                    st.metric(
                        "Direction Alignment",
                        "UNKNOWN"
                    )

            st.markdown(
                "**Why this location?**"
            )

            st.write(
                "The location is selected from eligible "
                "nearby GIS candidates using incident "
                "proximity, directional alignment, "
                "water proximity, and feature criticality."
            )

            basis = selected_next.get(
                "selection_basis"
            )

            if isinstance(
                basis,
                dict
            ) and basis:

                st.write(
                    "Selection basis:"
                )

                basis_items = []

                for key, value in basis.items():

                    label = str(
                        key
                    ).replace(
                        "_",
                        " "
                    ).title()

                    if isinstance(
                        value,
                        float
                    ):

                        value_text = (
                            f"{value:.3f}"
                        )

                    else:

                        value_text = str(
                            value
                        )

                    basis_items.append(
                        f"**{label}:** {value_text}"
                    )

                st.markdown(
                    "  \n".join(
                        basis_items
                    )
                )

        else:

            st.info(
                "No next verification location selected."
            )

        # ====================================================
        # INCIDENT VERIFICATION MAP
        #
        # IMPORTANT:
        # The map is intentionally inside the selected incident
        # block and directly follows Next Verification Location.
        #
        # No separate 'Incident Verification Map' title.
        # ====================================================

        try:

            gis_context = incident_data.get(
                "gis_context",
                {}
            )

            if not isinstance(
                gis_context,
                dict
            ):
                gis_context = {}

            verification_features = gis_context.get(
                "features",
                []
            )

            if not isinstance(
                verification_features,
                list
            ):
                verification_features = []

            if (
                center_lat is not None
                and center_lon is not None
            ):

                verification_map = folium.Map(
                    location=[
                        float(center_lat),
                        float(center_lon)
                    ],
                    zoom_start=15,
                    control_scale=True
                )

                # ------------------------------------------------
                # INCIDENT CENTER
                # ------------------------------------------------

                folium.Marker(
                    [
                        float(center_lat),
                        float(center_lon)
                    ],
                    tooltip=(
                        f"🔴 {incident_id} — Incident Center"
                    ),
                    popup=(
                        f"<b>🔴 Incident Center</b><br>"
                        f"<b>Incident:</b> {incident_id}<br>"
                        f"<b>Latitude:</b> {float(center_lat):.6f}<br>"
                        f"<b>Longitude:</b> {float(center_lon):.6f}"
                    ),
                    icon=folium.Icon(
                        color="red",
                        icon="warning-sign"
                    )
                ).add_to(
                    verification_map
                )

                # ------------------------------------------------
                # GIS FEATURE ICONS
                # ------------------------------------------------
                #
                # These are the SAME semantic categories used
                # by the original working verification map.
                #
                # School   -> green education
                # Hospital -> red plus
                # Bridge   -> orange road
                # Water    -> blue tint
                # Other    -> gray info-sign
                # ------------------------------------------------

                for feature in verification_features:

                    if not isinstance(
                        feature,
                        dict
                    ):
                        continue

                    # --------------------------------------------------------
                    # MAP DISPLAY ONLY — EXCLUDE ROADS
                    #
                    # Roads remain available to the GIS system if needed.
                    # They are simply not drawn on this verification map.
                    # --------------------------------------------------------

                    _map_feature_type = str(
                        feature.get(
                            "feature_type",
                            feature.get(
                                "type",
                                ""
                            )
                        )
                    ).strip().lower()

                    _map_feature_name = str(
                        feature.get(
                            "name",
                            feature.get(
                                "display_name",
                                feature.get(
                                    "location",
                                    ""
                                )
                            )
                        )
                    ).strip().lower()

                    _road_tokens = (
                        "road",
                        "street",
                        "lane",
                        "highway",
                        "junction",
                        "avenue",
                        "bypass",
                        "crossing",
                        "main road",
                        "link road",
                        "service road",
                    )

                    _is_map_road = (
                        any(
                            token in _map_feature_type
                            for token in _road_tokens
                        )
                        or
                        any(
                            token in _map_feature_name
                            for token in _road_tokens
                        )
                    )

                    if _is_map_road:
                        continue
                        continue

                    lat = feature.get(
                        "latitude"
                    )

                    lon = feature.get(
                        "longitude"
                    )

                    if (
                        lat is None
                        or lon is None
                    ):
                        continue

                    try:

                        lat = float(
                            lat
                        )

                        lon = float(
                            lon
                        )

                    except Exception:

                        continue

                    feature_type = str(
                        feature.get(
                            "feature_type",
                            "Other"
                        )
                    )

                    name = str(
                        feature.get(
                            "name",
                            "Unnamed"
                        )
                    )

                    distance = feature.get(
                        "distance_m"
                    )

                    # Normalize feature type.
                    feature_type_lower = (
                        feature_type
                        .strip()
                        .lower()
                    )

                    # ------------------------------------------------
                    # SCHOOL
                    # ------------------------------------------------

                    if (
                        feature_type_lower
                        in {
                            "school",
                            "schools",
                            "educational institution",
                            "education"
                        }
                    ):

                        marker_color = "green"
                        marker_icon = "education"
                        display_type = "School"

                    # ------------------------------------------------
                    # HOSPITAL
                    # ------------------------------------------------

                    elif (
                        feature_type_lower
                        in {
                            "hospital",
                            "hospitals",
                            "clinic",
                            "healthcare"
                        }
                    ):

                        marker_color = "red"
                        marker_icon = "plus"
                        display_type = "Hospital"

                    # ------------------------------------------------
                    # BRIDGE
                    # ------------------------------------------------

                    elif (
                        feature_type_lower
                        in {
                            "bridge",
                            "bridges"
                        }
                    ):

                        marker_color = "orange"
                        marker_icon = "road"
                        display_type = "Bridge"

                    # ------------------------------------------------
                    # WATER
                    # ------------------------------------------------

                    elif (
                        feature_type_lower
                        in {
                            "waterbody",
                            "water body",
                            "water",
                            "water resource",
                            "water resources",
                            "reservoir",
                            "reservoirs",
                            "lake",
                            "river",
                            "canal",
                            "pond"
                        }
                        or
                        "water" in feature_type_lower
                    ):

                        marker_color = "blue"
                        marker_icon = "tint"
                        display_type = "Water Resource"

                    # ------------------------------------------------
                    # OTHER USEFUL GIS FEATURES
                    # ------------------------------------------------

                    else:

                        marker_color = "gray"
                        marker_icon = "info-sign"
                        display_type = feature_type

                    distance_text = (
                        f"{float(distance):.1f} m"
                        if distance is not None
                        else "Unknown"
                    )

                    popup_text = (
                        f"<b>{name}</b><br>"
                        f"<b>Type:</b> {display_type}<br>"
                        f"<b>Distance:</b> {distance_text}"
                    )

                    folium.Marker(
                        [
                            lat,
                            lon
                        ],
                        tooltip=(
                            f"{display_type}: {name}"
                        ),
                        popup=popup_text,
                        icon=folium.Icon(
                            color=marker_color,
                            icon=marker_icon
                        )
                    ).add_to(
                        verification_map
                    )

                # ------------------------------------------------
                # NEXT VERIFICATION LOCATION
                # ------------------------------------------------
                #
                # This is drawn LAST so the purple star is visible
                # above other GIS markers when locations overlap.
                # ------------------------------------------------

                if selected_next:

                    next_lat = selected_next.get(
                        "latitude"
                    )

                    next_lon = selected_next.get(
                        "longitude"
                    )

                    if (
                        next_lat is not None
                        and next_lon is not None
                    ):

                        try:

                            next_lat = float(
                                next_lat
                            )

                            next_lon = float(
                                next_lon
                            )

                            next_name = selected_next.get(
                                "name",
                                "Unnamed location"
                            )

                            next_type = selected_next.get(
                                "feature_type",
                                "Unknown"
                            )

                            next_distance = selected_next.get(
                                "distance_m",
                                0
                            )

                            next_score = selected_next.get(
                                "priority_score",
                                0
                            )

                            folium.Marker(
                                [
                                    next_lat,
                                    next_lon
                                ],
                                tooltip=(
                                    "⭐ NEXT VERIFICATION — "
                                    f"{next_name}"
                                ),
                                popup=(
                                    "<b>⭐ NEXT VERIFICATION</b><br>"
                                    f"<b>Location:</b> {next_name}<br>"
                                    f"<b>Type:</b> {next_type}<br>"
                                    f"<b>Distance:</b> "
                                    f"{next_distance} m<br>"
                                    f"<b>Priority Score:</b> "
                                    f"{next_score}"
                                ),
                                icon=folium.Icon(
                                    color="purple",
                                    icon="star"
                                )
                            ).add_to(
                                verification_map
                            )

                        except Exception:
                            pass

                # ------------------------------------------------
                # MAP LEGEND
                # ------------------------------------------------

                legend_html = """
                <div style="
                    position: fixed;
                    bottom: 25px;
                    left: 25px;
                    z-index: 999999;
                    background-color: #ffffff;
                    color: #111111;
                    border: 2px solid #777777;
                    border-radius: 6px;
                    padding: 10px 12px;
                    font-size: 13px;
                    font-family: Arial, sans-serif;
                    line-height: 1.8;
                    font-weight: 600;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
                    min-width: 220px;
                    opacity: 1;
                    visibility: visible;
                    display: block;
                ">
                    <div style="
                        color: #111111;
                        font-weight: 700;
                        margin-bottom: 5px;
                    ">
                        CrisisWeave Verification
                    </div>
                    <div>🔴 Incident Center</div>
                    <div>⭐ Next Verification</div>
                    <div>🟢 School</div>
                    <div>✚ Hospital</div>
                    <div>🟠 Bridge</div>
                    <div>💧 Water Resource</div>
                    <div>⚪ Other GIS Feature</div>
                </div>
                """

                verification_map.get_root().html.add_child(
                    folium.Element(
                        legend_html
                    )
                )

                st_folium(
                    verification_map,
                    width=None,
                    height=600,
                    key=(
                        f"verification_map_"
                        f"{incident_id}"
                    ),
                    returned_objects=[],
                    use_container_width=True
                )

            else:

                st.warning(
                    "Incident coordinates are unavailable, "
                    "so the verification map cannot be rendered."
                )

        except Exception as map_exc:

            st.error(
                "Verification map could not be rendered."
            )

            st.exception(
                map_exc
            )

        # ====================================================
        # SPATIAL VERIFICATION
        # ====================================================

        st.markdown(
            "### Spatial Verification"
        )

        verification_status = verification.get(
            "status",
            "UNKNOWN"
        )

        verification_score = verification.get(
            "score"
        )

        if verification_score is None:

            verification_score = verification.get(
                "verification_score"
            )

        if verification_score is not None:

            try:

                verification_percent = (
                    float(
                        verification_score
                    ) * 100
                )

                verification_percent_text = (
                    f"{verification_percent:.1f}%"
                )

            except Exception:

                verification_percent_text = (
                    str(
                        verification_score
                    )
                )

        else:

            verification_percent_text = (
                "UNKNOWN"
            )

        vm1, vm2 = st.columns(
            2
        )

        with vm1:

            st.metric(
                "Verification Status",
                verification_status
            )

        with vm2:

            st.metric(
                "Verification Score",
                verification_percent_text
            )

        if center_lat is not None:

            st.write(
                "Incident latitude:",
                float(center_lat)
            )

        if center_lon is not None:

            st.write(
                "Incident longitude:",
                float(center_lon)
            )

        # ====================================================
        # OPERATIONAL RECOMMENDATION
        # ====================================================

        st.markdown(
            "### Operational Recommendation"
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
        # RECOMMENDATION EVIDENCE
        # ----------------------------------------------------

        evidence = recommendation.get(
            "evidence",
            {}
        )

        if isinstance(
            evidence,
            dict
        ) and evidence:

            st.markdown(
                "#### Recommendation Evidence"
            )

            ev1, ev2, ev3 = st.columns(
                3
            )

            with ev1:

                st.write(
                    "**Fused severity:**",
                    evidence.get(
                        "fused_severity",
                        "N/A"
                    )
                )

                st.write(
                    "**Latest severity:**",
                    evidence.get(
                        "latest_severity",
                        "N/A"
                    )
                )

                st.write(
                    "**Temporal trend:**",
                    evidence.get(
                        "temporal_trend",
                        "N/A"
                    )
                )

            with ev2:

                fusion_conf = evidence.get(
                    "fusion_confidence"
                )

                if fusion_conf is not None:

                    try:

                        fusion_conf_text = (
                            f"{float(fusion_conf) * 100:.1f}%"
                        )

                    except Exception:

                        fusion_conf_text = str(
                            fusion_conf
                        )

                else:

                    fusion_conf_text = "N/A"

                st.write(
                    "**Fusion confidence:**",
                    fusion_conf_text
                )

                st.write(
                    "**Uncertainty:**",
                    evidence.get(
                        "uncertainty",
                        "N/A"
                    )
                )

                st.write(
                    "**Verification:**",
                    evidence.get(
                        "verification_status",
                        verification_status
                    )
                )

            with ev3:

                rec_score = evidence.get(
                    "verification_score"
                )

                if rec_score is not None:

                    try:

                        rec_score_text = (
                            f"{float(rec_score) * 100:.1f}%"
                        )

                    except Exception:

                        rec_score_text = str(
                            rec_score
                        )

                else:

                    rec_score_text = "N/A"

                st.write(
                    "**Verification score:**",
                    rec_score_text
                )

                st.write(
                    "**Conflict status:**",
                    evidence.get(
                        "conflict_status",
                        "N/A"
                    )
                )

        # ====================================================
        # INCIDENT OBSERVATIONS
        # ====================================================

        incident_observations = incident_data.get(
            "observations",
            []
        )

        if incident_observations:

            with st.expander(
                f"View {incident_id} observations"
            ):

                observation_rows = []

                for obs in incident_observations:

                    if not isinstance(
                        obs,
                        dict
                    ):
                        continue

                    prediction = obs.get(
                        "prediction",
                        {}
                    )

                    if not isinstance(
                        prediction,
                        dict
                    ):
                        prediction = {}

                    observation_rows.append(
                        {
                            "Observation ID":
                                obs.get(
                                    "observation_id",
                                    "N/A"
                                ),

                            "Timestamp":
                                obs.get(
                                    "timestamp"
                                )
                                or obs.get(
                                    "observed_at"
                                )
                                or "—",

                            "Latitude":
                                obs.get(
                                    "latitude",
                                    "N/A"
                                ),

                            "Longitude":
                                obs.get(
                                    "longitude",
                                    "N/A"
                                ),

                            "Severity":
                                prediction.get(
                                    "severity"
                                ) or "—",

                            "Category":
                                prediction.get(
                                    "category"
                                ) or "—"
                        }
                    )

                if observation_rows:

                    st.dataframe(
                        pd.DataFrame(
                            observation_rows
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

    else:

        st.warning(
            "No incident results are available."
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "CrisisWeave Live — runtime evidence reconstruction. "
    "Saved result files are not used as analysis inputs."
)



