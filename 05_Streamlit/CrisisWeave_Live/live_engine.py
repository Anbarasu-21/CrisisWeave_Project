import os
import math
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
PROJECT_ROOT = '/content/gdrive/MyDrive/CrisisWeave_Project'
CNN_PATH = os.path.join(PROJECT_ROOT, '03_Models', 'CNN', 'best_model.keras')
BERT_PATH = os.path.join(PROJECT_ROOT, '03_Models', 'BERT', 'HumAID_Flood_BERT')
CNN_CLASS_NAMES = {0: 'High', 1: 'Low', 2: 'Moderate'}
BERT_CLASS_NAMES = {0: 'caution_and_advice', 1: 'displaced_people_and_evacuations', 2: 'infrastructure_and_utility_damage', 3: 'injured_or_dead_people', 4: 'missing_or_found_people', 5: 'not_humanitarian', 6: 'other_relevant_information', 7: 'requests_or_urgent_needs', 8: 'rescue_volunteering_or_donation_effort', 9: 'sympathy_and_support'}
SEVERITY_VALUE = {'Low': 0, 'Moderate': 1, 'High': 2}
SOURCE_BASELINE = {'CNN': 0.75, 'BERT': 0.75, 'OSM': 0.9}
DEFAULT_DECAY_RATE = 0.25

def load_cnn():
    return tf.keras.models.load_model(CNN_PATH, compile=False)

def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    model.eval()
    return (tokenizer, model)

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def parse_datetime(value):
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))

def observation_id():
    raw = f'{datetime.now().isoformat()}_{np.random.random()}'
    return 'OBS_' + hashlib.sha1(raw.encode()).hexdigest()[:10].upper()

def predict_cnn(model, image_bytes):
    image = Image.open(__import__('io').BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    severity = CNN_CLASS_NAMES[class_id]
    confidence = float(probabilities[class_id])
    return {'class_id': class_id, 'severity': severity, 'confidence': confidence, 'probabilities': {CNN_CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(probabilities))}}

def predict_bert(tokenizer, model, text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        output = model(**encoded)
    probabilities = torch.softmax(output.logits, dim=-1)[0].cpu().numpy()
    class_id = int(np.argmax(probabilities))
    category = BERT_CLASS_NAMES[class_id]
    confidence = float(probabilities[class_id])
    return {'class_id': class_id, 'category': category, 'confidence': confidence, 'probabilities': {BERT_CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(probabilities))}}

def create_image_observation(image_bytes, filename, observed_at, received_at, latitude, longitude, cnn_result):
    return {'observation_id': observation_id(), 'source': 'CNN', 'modality': 'image', 'filename': filename, 'observed_at': parse_datetime(observed_at).isoformat(), 'received_at': parse_datetime(received_at).isoformat(), 'latitude': safe_float(latitude), 'longitude': safe_float(longitude), 'prediction': {'severity': cnn_result['severity'], 'class_id': cnn_result['class_id'], 'confidence': cnn_result['confidence'], 'probabilities': cnn_result['probabilities']}, 'image_bytes': image_bytes}

def create_text_observation(text, observed_at, received_at, latitude, longitude, bert_result):
    return {'observation_id': observation_id(), 'source': 'BERT', 'modality': 'text', 'text': text, 'observed_at': parse_datetime(observed_at).isoformat(), 'received_at': parse_datetime(received_at).isoformat(), 'latitude': safe_float(latitude), 'longitude': safe_float(longitude), 'prediction': {'category': bert_result['category'], 'class_id': bert_result['class_id'], 'confidence': bert_result['confidence'], 'probabilities': bert_result['probabilities']}}

def freshness_score(observed_at, reference_time, decay_rate=DEFAULT_DECAY_RATE):
    observed = parse_datetime(observed_at)
    reference = parse_datetime(reference_time)
    age_hours = max(0.0, (reference - observed).total_seconds() / 3600.0)
    freshness = math.exp(-decay_rate * age_hours)
    return {'age_hours': age_hours, 'freshness': clamp(freshness)}

def haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Calculate geographic distance between two latitude/longitude
    coordinates using the Haversine formula.

    Returns:
        Distance in meters.
    """
    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)

    if None in (lat1, lon1, lat2, lon2):
        return None

    earth_radius_m = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2.0) ** 2
    )

    return 2.0 * earth_radius_m * math.atan2(
        math.sqrt(a),
        math.sqrt(max(0.0, 1.0 - a))
    )


def group_observations_by_incident(
    observations,
    incident_radius_m=100.0
):
    """
    Group observations into independent geographic incidents.

    Important:
    - Uses observation coordinates.
    - Uses a stable incident center for each group.
    - Does NOT use transitive chaining.
    - Different geographic clusters remain separate.
    """

    valid = []

    for observation in observations:
        lat = safe_float(observation.get("latitude"))
        lon = safe_float(observation.get("longitude"))

        if lat is None or lon is None:
            continue

        item = dict(observation)
        item["_group_lat"] = lat
        item["_group_lon"] = lon
        valid.append(item)

    if not valid:
        return []

    valid = sort_observations(valid)

    groups = []

    for observation in valid:

        lat = observation["_group_lat"]
        lon = observation["_group_lon"]

        assigned_group = None
        assigned_distance = None

        for group in groups:

            distance = haversine_distance_m(
                lat,
                lon,
                group["center_lat"],
                group["center_lon"]
            )

            if (
                distance is not None
                and distance <= incident_radius_m
            ):
                assigned_group = group
                assigned_distance = distance
                break

        if assigned_group is None:

            group_id = f"INCIDENT-{len(groups) + 1:03d}"

            groups.append(
                {
                    "incident_id": group_id,
                    "center_lat": lat,
                    "center_lon": lon,
                    "observations": [observation],
                    "observation_count": 1,
                    "max_observation_distance_m": 0.0
                }
            )

        else:

            assigned_group["observations"].append(
                observation
            )

            assigned_group["observation_count"] += 1

            if assigned_distance is not None:
                assigned_group[
                    "max_observation_distance_m"
                ] = max(
                    assigned_group[
                        "max_observation_distance_m"
                    ],
                    assigned_distance
                )

    for group in groups:

        for observation in group["observations"]:

            observation.pop(
                "_group_lat",
                None
            )

            observation.pop(
                "_group_lon",
                None
            )

    return groups


def sort_observations(observations):
    return sorted(observations, key=lambda x: parse_datetime(x['observed_at']))

def image_observations(observations):
    return [x for x in observations if x.get('source') == 'CNN']

def text_observations(observations):
    return [x for x in observations if x.get('source') == 'BERT']

def severity_trend(observations):
    images = image_observations(sort_observations(observations))
    if len(images) < 2:
        return {'trend': 'INSUFFICIENT_DATA', 'initial_severity': images[0]['prediction']['severity'] if images else None, 'latest_severity': images[-1]['prediction']['severity'] if images else None}
    values = [SEVERITY_VALUE[item['prediction']['severity']] for item in images]
    initial = values[0]
    latest = values[-1]
    if latest > initial:
        trend = 'INCREASING'
    elif latest < initial:
        trend = 'DECREASING'
    elif len(values) >= 3:
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        if slope > 0.15:
            trend = 'INCREASING'
        elif slope < -0.15:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'
    else:
        trend = 'STABLE'
    return {'trend': trend, 'initial_severity': images[0]['prediction']['severity'], 'latest_severity': images[-1]['prediction']['severity']}

def progression_direction(observations):
    ordered = sort_observations(observations)
    points = []
    for item in ordered:
        lat = safe_float(item.get('latitude'))
        lon = safe_float(item.get('longitude'))
        if lat is not None and lon is not None:
            points.append((lat, lon))
    if len(points) < 2:
        return 'UNKNOWN'
    lat1, lon1 = points[0]
    lat2, lon2 = points[-1]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    if abs(dlat) < 1e-06 and abs(dlon) < 1e-06:
        return 'STATIONARY'
    angle = math.degrees(math.atan2(dlon, dlat))
    if angle < 0:
        angle += 360
    directions = [(0, 'N'), (45, 'NE'), (90, 'E'), (135, 'SE'), (180, 'S'), (225, 'SW'), (270, 'W'), (315, 'NW')]
    nearest = min(directions, key=lambda item: min(abs(angle - item[0]), 360 - abs(angle - item[0])))
    return nearest[1]

def build_temporal_state(observations, analysis_time=None):
    if analysis_time is None:
        analysis_time = utc_now_iso()
    ordered = sort_observations(observations)
    if not ordered:
        return {'observation_count': 0, 'trend': 'NO_DATA', 'progression_direction': 'UNKNOWN', 'initial_severity': None, 'latest_severity': None, 'latest_observation': None}
    trend_info = severity_trend(ordered)
    enriched = []
    for item in ordered:
        freshness = freshness_score(item['observed_at'], analysis_time)
        enriched.append({'observation_id': item['observation_id'], 'source': item['source'], 'observed_at': item['observed_at'], 'freshness': freshness['freshness'], 'age_hours': freshness['age_hours']})
    return {'observation_count': len(ordered), 'first_observation': ordered[0]['observed_at'], 'latest_observation': ordered[-1]['observed_at'], 'initial_severity': trend_info['initial_severity'], 'latest_severity': trend_info['latest_severity'], 'trend': trend_info['trend'], 'progression_direction': progression_direction(ordered), 'freshness': enriched}

def severity_difference(severity_a, severity_b):
    return abs(SEVERITY_VALUE[severity_a] - SEVERITY_VALUE[severity_b])

def detect_conflict(observations, analysis_time=None, conflict_window_hours=6):
    images = image_observations(sort_observations(observations))
    if len(images) < 2:
        return {'status': 'INSUFFICIENT_DATA', 'overall_conflict': None, 'difference': None, 'temporal_tension': 'INSUFFICIENT_DATA', 'humanitarian_evidence': 'AVAILABLE' if text_observations(observations) else 'NONE'}
    if analysis_time is None:
        analysis_time = utc_now_iso()
    latest = images[-1]
    comparable = []
    latest_time = parse_datetime(latest['observed_at'])
    for item in images[:-1]:
        item_time = parse_datetime(item['observed_at'])
        hours = abs((latest_time - item_time).total_seconds() / 3600.0)
        if hours <= conflict_window_hours:
            comparable.append(item)
    if not comparable:
        return {'status': 'NO_RECENT_COMPARISON', 'overall_conflict': False, 'difference': None, 'temporal_tension': 'NO_RECENT_CONFLICT_EVIDENCE', 'humanitarian_evidence': 'SUPPORTING' if text_observations(observations) else 'NONE'}
    previous = comparable[-1]
    current_severity = latest['prediction']['severity']
    previous_severity = previous['prediction']['severity']
    difference = severity_difference(current_severity, previous_severity)
    if difference == 0:
        status = 'AGREEMENT'
        overall = False
    elif difference == 1:
        status = 'MINOR'
        overall = True
    else:
        status = 'MAJOR'
        overall = True
    temporal = 'CONSISTENT' if difference <= 1 else 'TENSION'
    return {'status': status, 'overall_conflict': overall, 'difference': difference, 'previous_severity': previous_severity, 'latest_severity': current_severity, 'temporal_tension': temporal, 'humanitarian_evidence': 'SUPPORTING' if text_observations(observations) else 'NONE'}

def observation_reliability(observation, analysis_time, conflict_penalty=0.0):
    source = observation['source']
    baseline = SOURCE_BASELINE.get(source, 0.7)
    freshness = freshness_score(observation['observed_at'], analysis_time)
    prediction = observation.get('prediction', {})
    confidence = safe_float(prediction.get('confidence'))
    if confidence is None:
        confidence = 1.0
    reliability = baseline * freshness['freshness'] * confidence * (1.0 - conflict_penalty)
    return {'observation_id': observation['observation_id'], 'source': source, 'baseline': baseline, 'freshness': freshness['freshness'], 'confidence': confidence, 'conflict_penalty': conflict_penalty, 'reliability': clamp(reliability)}

def build_reliability_state(observations, conflict, analysis_time=None):
    if analysis_time is None:
        analysis_time = utc_now_iso()
    status = conflict.get('status', 'UNKNOWN')
    penalty_map = {'AGREEMENT': 0.0, 'MINOR': 0.25, 'MAJOR': 0.5}
    penalty = penalty_map.get(status, 0.0)
    records = [observation_reliability(item, analysis_time, penalty) for item in observations]
    return {'records': records, 'mean_reliability': float(np.mean([x['reliability'] for x in records])) if records else 0.0}

def fuse_severity(observations, reliability_state, temporal_state):
    images = image_observations(observations)
    if not images:
        return {'fused_severity': None, 'confidence': 0.0, 'uncertainty': 'HIGH'}
    reliability_lookup = {item['observation_id']: item['reliability'] for item in reliability_state['records']}
    scores = {'Low': 0.0, 'Moderate': 0.0, 'High': 0.0}
    total_weight = 0.0
    for item in images:
        severity = item['prediction']['severity']
        weight = reliability_lookup.get(item['observation_id'], 0.0)
        scores[severity] += weight
        total_weight += weight
    if total_weight <= 0:
        return {'fused_severity': None, 'confidence': 0.0, 'uncertainty': 'HIGH'}
    normalized = {key: value / total_weight for key, value in scores.items()}
    fused = max(normalized, key=normalized.get)
    confidence = normalized[fused]
    latest = temporal_state.get('latest_severity')
    if latest == fused:
        confidence = min(1.0, confidence + 0.05)
    if confidence >= 0.75:
        uncertainty = 'LOW'
    elif confidence >= 0.5:
        uncertainty = 'MEDIUM'
    else:
        uncertainty = 'HIGH'
    return {'fused_severity': fused, 'confidence': round(confidence, 4), 'uncertainty': uncertainty, 'class_scores': normalized}

def humanitarian_state(observations):
    texts = text_observations(sort_observations(observations))
    if not texts:
        return {'latest_category': None, 'latest_confidence': 0.0, 'observation_count': 0}
    latest = texts[-1]
    return {'latest_category': latest['prediction']['category'], 'latest_confidence': latest['prediction']['confidence'], 'observation_count': len(texts)}

def fetch_live_gis(latitude, longitude, radius_m):
    """
    Retrieve live OSM features around a runtime incident.

    Coordinate policy:
    -------------------
    - Original OSM geometries remain in EPSG:4326.
    - Latitude/longitude are extracted from those original
      WGS84 geometries.
    - A separate projected UTM GeoDataFrame is used ONLY
      for metric distance calculations.

    This prevents projected coordinates from being mistaken
    for latitude/longitude.
    """
    import osmnx as ox
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    latitude = float(latitude)
    longitude = float(longitude)
    radius_m = float(radius_m)
    center_point = (latitude, longitude)
    tags = {'amenity': ['hospital', 'school'], 'healthcare': True, 'highway': True, 'bridge': True, 'man_made': ['bridge'], 'building': ['school'], 'natural': ['water'], 'waterway': True, 'water': True, 'landuse': ['reservoir', 'basin', 'salt_pond']}
    gdf = ox.features_from_point(center_point, tags=tags, dist=radius_m)
    if gdf is None or len(gdf) == 0:
        return {'incident': {'latitude': latitude, 'longitude': longitude}, 'radius_m': radius_m, 'features': [], 'counts': {'roads': 0, 'hospitals': 0, 'schools': 0, 'bridges': 0, 'waterbodies': 0}, 'nearest_road': None, 'nearest_hospital': None, 'nearest_school': None, 'nearest_bridge': None, 'nearest_waterbody': None}
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.to_crs('EPSG:4326')
    gdf_wgs84 = gdf.copy()
    metric_crs = gdf_wgs84.estimate_utm_crs()
    if metric_crs is None:
        raise RuntimeError('Could not determine a suitable UTM CRS')
    gdf_metric = gdf_wgs84.to_crs(metric_crs)
    center_wgs84 = Point(longitude, latitude)
    center_metric = gpd.GeoSeries([center_wgs84], crs='EPSG:4326').to_crs(metric_crs).iloc[0]

    def value(row, key):
        try:
            result = row.get(key, None)
        except Exception:
            result = None
        if result is None:
            return None
        try:
            if pd.isna(result):
                return None
        except Exception:
            pass
        return result

    def text(row, key):
        result = value(row, key)
        if result is None:
            return ''
        return str(result).strip().lower()

    def has_tag(row, key):
        result = value(row, key)
        if result is None:
            return False
        if isinstance(result, bool):
            return result
        return str(result).strip().lower() not in {'', 'nan', 'none', 'false', 'no', '0'}
    records = []
    seen = set()
    for idx in gdf_wgs84.index:
        row = gdf_wgs84.loc[idx]
        metric_row = gdf_metric.loc[idx]
        geometry_wgs84 = row.geometry
        geometry_metric = metric_row.geometry
        if geometry_wgs84 is None or geometry_wgs84.is_empty or geometry_metric is None or geometry_metric.is_empty:
            continue
        if isinstance(idx, tuple):
            osm_key = tuple((str(x) for x in idx))
        else:
            osm_key = (str(idx),)
        natural = text(row, 'natural')
        water = text(row, 'water')
        waterway = text(row, 'waterway')
        landuse = text(row, 'landuse')
        amenity = text(row, 'amenity')
        healthcare = text(row, 'healthcare')
        highway = value(row, 'highway')
        is_water = natural == 'water' or bool(water) or bool(waterway) or (landuse in {'reservoir', 'basin', 'salt_pond'})
        is_bridge = (
            has_tag(row, 'bridge')
            or text(row, 'man_made') == 'bridge'
        )
        is_hospital = amenity == 'hospital' or healthcare == 'hospital'
        is_school = (
            amenity == 'school'
            or text(row, 'building') == 'school'
        )
        is_road = highway is not None
        if is_water:
            feature_type = 'Waterbody'
        elif is_bridge:
            feature_type = 'Bridge'
        elif is_hospital:
            feature_type = 'Hospital'
        elif is_school:
            feature_type = 'School'
        elif is_road:
            feature_type = 'Road'
        else:
            continue
        unique_key = (feature_type, osm_key)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        representative_metric = geometry_metric.representative_point()
        distance_m = float(representative_metric.distance(center_metric))
        representative_wgs84 = geometry_wgs84.representative_point()
        feature_lat = float(representative_wgs84.y)
        feature_lon = float(representative_wgs84.x)
        name = value(row, 'name')
        if name is None:
            name = 'Unnamed'
        relationship = None
        if feature_type == 'Waterbody':
            if distance_m <= 50:
                relationship = 'Adjacent Waterbody'
            elif distance_m <= 200:
                relationship = 'Near Waterbody'
            else:
                relationship = 'Far Waterbody'
        records.append({'feature_type': feature_type, 'name': str(name), 'distance_m': round(distance_m, 3), 'latitude': feature_lat, 'longitude': feature_lon, 'relationship': relationship, 'osm_id': str(osm_key)})
    records.sort(key=lambda x: x['distance_m'])
    counts = {'roads': sum((x['feature_type'] == 'Road' for x in records)), 'hospitals': sum((x['feature_type'] == 'Hospital' for x in records)), 'schools': sum((x['feature_type'] == 'School' for x in records)), 'bridges': sum((x['feature_type'] == 'Bridge' for x in records)), 'waterbodies': sum((x['feature_type'] == 'Waterbody' for x in records))}

    def nearest(feature_type):
        matches = [x for x in records if x['feature_type'] == feature_type]
        if not matches:
            return None
        return matches[0]
    return {'incident': {'latitude': latitude, 'longitude': longitude}, 'radius_m': radius_m, 'features': records, 'counts': counts, 'nearest_road': nearest('Road'), 'nearest_hospital': nearest('Hospital'), 'nearest_school': nearest('School'), 'nearest_bridge': nearest('Bridge'), 'nearest_waterbody': nearest('Waterbody')}

def nearest_feature(gis_context, feature_type):
    candidates = [item for item in gis_context.get('features', []) if item['feature_type'] == feature_type]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x['distance_m'])

def verification_score(gis_context):
    score = 0.0
    evidence = []
    weights = {'Road': 0.15, 'Waterbody': 0.25, 'Hospital': 0.2, 'School': 0.15, 'Bridge': 0.15}
    thresholds = {'Road': 100, 'Waterbody': 200, 'Hospital': 500, 'School': 500, 'Bridge': 500}
    for feature_type in weights:
        item = nearest_feature(gis_context, feature_type)
        if item is None:
            continue
        distance = item['distance_m']
        if distance <= thresholds[feature_type]:
            score += weights[feature_type]
            evidence.append({'feature': feature_type, 'distance_m': distance, 'contribution': weights[feature_type]})
    if len(evidence) >= 3:
        score += 0.1
    score = min(score, 1.0)
    if score >= 0.75:
        status = 'VERIFIED'
    elif score >= 0.5:
        status = 'PARTIALLY VERIFIED'
    else:
        status = 'UNVERIFIED'
    return {'status': status, 'score': round(score, 3), 'supporting_spatial_evidence': evidence}
CRITICALITY = {'Hospital': 1.0, 'Bridge': 0.9, 'School': 0.75, 'Road': 0.6, 'Waterbody': 0.5}

def bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    angle = math.degrees(math.atan2(y, x))
    return (angle + 360) % 360

def direction_alignment(target_bearing, progression):
    mapping = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
    if progression not in mapping:
        return 0.5
    target = mapping[progression]
    diff = abs(target_bearing - target)
    diff = min(diff, 360 - diff)
    if diff <= 22.5:
        return 1.0
    if diff <= 45:
        return 0.8
    if diff <= 90:
        return 0.5
    if diff <= 135:
        return 0.2
    return 0.0

def distance_score(distance_m):
    if distance_m <= 50:
        return 1.0
    if distance_m <= 100:
        return 0.8
    if distance_m <= 250:
        return 0.6
    if distance_m <= 500:
        return 0.4
    if distance_m <= 1000:
        return 0.2
    return 0.0

def build_dynamic_state(fused, temporal, humanitarian, conflict, reliability, gis_context, verification):
    fused_severity = fused.get('fused_severity')
    latest_severity = temporal.get('latest_severity')
    severity = fused_severity
    if latest_severity == 'High' and fused_severity != 'High':
        operational_risk = 'URGENT_VERIFICATION'
    elif fused_severity == 'High':
        operational_risk = 'CRITICAL'
    elif fused_severity == 'Moderate':
        operational_risk = 'ELEVATED'
    elif fused_severity == 'Low':
        operational_risk = 'LOW'
    else:
        operational_risk = 'UNKNOWN'
    return {'current_state': {'severity': severity, 'fused_severity': fused_severity, 'latest_severity': latest_severity, 'confidence': fused.get('confidence', 0.0), 'uncertainty': fused.get('uncertainty', 'HIGH')}, 'humanitarian_state': humanitarian, 'temporal_state': temporal, 'conflict_state': conflict, 'geographic_state': gis_context, 'verification': verification, 'evidence_quality': {'mean_reliability': reliability.get('mean_reliability', 0.0)}, 'operational_risk': operational_risk}

def _cw_original_rank_priority_locations(
    gis_context,
    incident_lat,
    incident_lon,
    progression_direction,
    fused_severity,
    verification_status,
    latest_severity
):

    candidates = []

    latest_high_conflict = (
        latest_severity == "High"
        and fused_severity != "High"
    )

    eligible = (
        (
            fused_severity == "High"
            or latest_high_conflict
        )
        and verification_status
        in {
            "VERIFIED",
            "PARTIALLY VERIFIED"
        }
    )

    features = gis_context.get(
        "features",
        []
    )

    for item in features:

        if item.get("feature_type") not in {
            "Hospital",
            "Bridge",
            "School"
        }:
            continue

        lat = item.get("latitude")
        lon = item.get("longitude")

        if lat is None or lon is None:
            continue

        distance_m = item.get(
            "distance_m",
            0
        )

        # --------------------------------------------------------
        # SPATIAL PROXIMITY
        # --------------------------------------------------------

        incident_proximity = distance_score(
            distance_m
        )

        # --------------------------------------------------------
        # PROGRESSION DIRECTION
        # --------------------------------------------------------

        b = bearing(
            incident_lat,
            incident_lon,
            lat,
            lon
        )

        align = direction_alignment(
            b,
            progression_direction
        )

        # --------------------------------------------------------
        # WATER CONTEXT
        # --------------------------------------------------------

        water = nearest_feature(
            gis_context,
            "Waterbody"
        )

        if water is not None:

            water_distance = abs(
                distance_m
                - water["distance_m"]
            )

            water_score = (
                1.0
                if water_distance <= 50
                else 0.8
                if water_distance <= 100
                else 0.5
                if water_distance <= 200
                else 0.0
            )

        else:

            water_distance = None
            water_score = 0.0

        # --------------------------------------------------------
        # FACILITY CRITICALITY
        # --------------------------------------------------------

        criticality = CRITICALITY.get(
            item["feature_type"],
            0.0
        )

        # --------------------------------------------------------
        # PRIORITY SCORE
        #
        # Proximity is intentionally dominant.
        #
        # 50% proximity
        # 20% progression alignment
        # 20% criticality
        # 10% water context
        # --------------------------------------------------------

        score = (
            0.50 * incident_proximity
            +
            0.20 * align
            +
            0.20 * criticality
            +
            0.10 * water_score
        )

        candidates.append({

            "name":
                item.get(
                    "name",
                    "Unnamed"
                ),

            "feature_type":
                item["feature_type"],

            "latitude":
                lat,

            "longitude":
                lon,

            "distance_m":
                distance_m,

            "direction_alignment":
                round(
                    align,
                    3
                ),

            "incident_proximity":
                round(
                    incident_proximity,
                    3
                ),

            "water_proximity":
                round(
                    water_score,
                    3
                ),

            "water_distance_m":
                water_distance,

            "base_criticality":
                criticality,

            "priority_score":
                round(
                    score,
                    3
                )
        })

    # ------------------------------------------------------------
    # CLOSEST / MOST RELEVANT CANDIDATES FIRST
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # NEXT VERIFICATION LOCATION
    #
    # Nearest eligible useful infrastructure first.
    # Existing priority score is retained as the tie-breaker.
    #
    # Roads are already excluded from the candidate set.
    # ------------------------------------------------------------

    candidates = list(enumerate(candidates))

    candidates.sort(
        key=lambda pair: (
            float(
                pair[1].get(
                    "distance_m",
                    999999999.0
                )
                if pair[1].get(
                    "distance_m",
                    None
                ) is not None
                else 999999999.0
            ),
            -float(
                pair[1].get(
                    "priority_score",
                    0.0
                )
                if pair[1].get(
                    "priority_score",
                    None
                ) is not None
                else 0.0
            ),
            pair[0]
        )
    )

    candidates = [
        item
        for _, item in candidates
    ]

    return {

        "eligible":
            eligible,

        "ranked_candidates":
            candidates,

        "selected_next_priority":
            (
                candidates[0]
                if eligible and candidates
                else None
            ),

        "latest_high_conflict":
            latest_high_conflict
    }


def build_recommendation(fused, temporal, verification, priority, conflict):
    severity = fused.get('fused_severity')
    latest_severity = temporal.get('latest_severity')
    verification_status = verification.get('status')
    verification_score = verification.get('score', 0.0)
    trend = temporal.get('trend')
    conflict_status = conflict.get('status')
    uncertainty = fused.get('uncertainty', 'HIGH')
    confidence = fused.get('confidence', 0.0)
    latest_high_conflict = latest_severity == 'High' and severity != 'High'
    if latest_high_conflict:
        decision = 'URGENT VERIFICATION REQUIRED'
        explanation = f'Latest evidence indicates HIGH severity, while reliability-weighted fusion indicates {severity} severity with {uncertainty} uncertainty. The disagreement should be resolved through additional spatial or field verification before an operational response decision.'
    elif severity == 'High':
        if verification_status not in {'VERIFIED', 'PARTIALLY VERIFIED'}:
            decision = 'VERIFY LOCATION BEFORE RESPONSE'
            explanation = 'The fused severity is HIGH, but the spatial verification status is insufficient to support an operational priority decision.'
        elif trend == 'INCREASING':
            decision = 'IMMEDIATE RESPONSE'
            explanation = 'The fused severity is HIGH, spatial context is sufficiently verified, and the temporal trend is INCREASING, indicating active escalation.'
        else:
            decision = 'HIGH PRIORITY RESPONSE'
            explanation = 'The fused severity is HIGH and spatial context is sufficiently verified, but the temporal trend does not indicate active escalation.'
    elif severity == 'Moderate':
        decision = 'MONITOR AND VERIFY'
        explanation = 'The fused severity is MODERATE. Continued observation and spatial verification are appropriate before escalation.'
    elif severity == 'Low':
        decision = 'CONTINUE MONITORING'
        explanation = 'The fused severity is LOW, so continued monitoring is appropriate unless new evidence indicates escalation.'
    else:
        decision = 'INSUFFICIENT EVIDENCE'
        explanation = 'Current observations are insufficient to reliably reconstruct the disaster state.'
    evidence = {'fused_severity': severity, 'latest_severity': latest_severity, 'fusion_confidence': round(confidence, 3), 'uncertainty': uncertainty, 'temporal_trend': trend, 'verification_status': verification_status, 'verification_score': round(verification_score, 3), 'conflict_status': conflict_status}
    return {'decision': decision, 'explanation': explanation, 'evidence': evidence, 'next_location': priority.get('selected_next_priority'), 'conflict_status': conflict_status, 'latest_severity': latest_severity, 'fused_severity': severity, 'latest_fused_conflict': latest_high_conflict}

def reconstruct_state(observations, gis_context, analysis_time=None):
    if analysis_time is None:
        analysis_time = utc_now_iso()
    temporal = build_temporal_state(observations, analysis_time)
    conflict = detect_conflict(observations, analysis_time)
    reliability = build_reliability_state(observations, conflict, analysis_time)
    fused = fuse_severity(observations, reliability, temporal)
    humanitarian = humanitarian_state(observations)
    verification = verification_score(gis_context)
    priority = rank_priority_locations(gis_context, gis_context['incident']['latitude'], gis_context['incident']['longitude'], temporal['progression_direction'], fused['fused_severity'], verification['status'], temporal['latest_severity'])
    dynamic_state = build_dynamic_state(fused, temporal, humanitarian, conflict, reliability, gis_context, verification)
    recommendation = build_recommendation(fused, temporal, verification, priority, conflict)
    return {'analysis_time': analysis_time, 'observation_count': len(observations), 'temporal': temporal, 'conflict': conflict, 'reliability': reliability, 'fusion': fused, 'humanitarian': humanitarian, 'verification': verification, 'dynamic_state': dynamic_state, 'priority': priority, 'recommendation': recommendation}

# ==============================================================================
# CRISISWEAVE — DISTANCE-FIRST VERIFICATION WRAPPER
# ==============================================================================

def rank_priority_locations(
    gis_context,
    incident_lat,
    incident_lon,
    progression_direction,
    fused_severity,
    verification_status,
    latest_severity
):
    """
    Preserve the original GIS ranking implementation while enforcing
    CrisisWeave's operational verification rule:

        USEFUL INFRASTRUCTURE > ROADS/STREETS
        DISTANCE FIRST
        CRITICALITY AS TIE-BREAKER

    The original function remains untouched as:
        _cw_original_rank_priority_locations()
    """

    # --------------------------------------------------------------------------
    # Run the original engine.
    # --------------------------------------------------------------------------

    original_result = _cw_original_rank_priority_locations(
        gis_context,
        incident_lat,
        incident_lon,
        progression_direction,
        fused_severity,
        verification_status,
        latest_severity
    )

    # --------------------------------------------------------------------------
    # Defensive handling.
    # --------------------------------------------------------------------------

    if original_result is None:
        return []

    if not isinstance(original_result, (list, tuple)):
        return original_result

    candidates = list(original_result)

    if not candidates:
        return []

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def text_value(item, *keys):
        for key in keys:
            try:
                value = item.get(key)
            except Exception:
                value = None

            if value is not None:
                return str(value).strip()

        return ""

    def number_value(item, *keys, default=999999999.0):
        for key in keys:
            try:
                value = item.get(key)
            except Exception:
                value = None

            if value is None:
                continue

            try:
                return float(value)
            except Exception:
                pass

        return default

    def feature_type(item):
        return text_value(
            item,
            "type",
            "feature_type",
            "category",
            "feature_class",
            "class",
            "amenity_type"
        ).lower()

    def feature_name(item):
        return text_value(
            item,
            "name",
            "location",
            "display_name",
            "feature_name",
            "title"
        ).lower()

    def distance(item):
        return number_value(
            item,
            "distance_m",
            "distance",
            "dist_m",
            "distanceMeters",
            default=999999999.0
        )

    def criticality(item):
        return number_value(
            item,
            "criticality",
            "base_criticality",
            "feature_criticality",
            default=0.0
        )

    # --------------------------------------------------------------------------
    # REMOVE ROADS / STREETS FROM NEXT-VERIFICATION CANDIDATES
    #
    # Roads are useful GIS context, but they should NOT become the selected
    # "Next Verification Location".
    # --------------------------------------------------------------------------

    road_words = (
        "road",
        "street",
        "lane",
        "highway",
        "junction",
        "avenue",
        "bypass",
        "cross",
        "main road",
        "link road",
        "service road"
    )

    useful_types = (
        "school",
        "hospital",
        "bridge",
        "water",
        "river",
        "lake",
        "reservoir",
        "dam",
        "clinic",
        "shelter",
        "fire",
        "police",
        "health",
        "emergency"
    )

    useful = []
    other_nonroad = []

    for item in candidates:

        if not isinstance(item, dict):
            continue

        ftype = feature_type(item)
        fname = feature_name(item)

        is_road = (
            any(word in ftype for word in road_words)
            or any(word in fname for word in road_words)
        )

        if is_road:
            continue

        is_useful = (
            any(word in ftype for word in useful_types)
            or any(word in fname for word in useful_types)
        )

        if is_useful:
            useful.append(item)
        else:
            other_nonroad.append(item)

    # --------------------------------------------------------------------------
    # Prefer recognised useful infrastructure.
    #
    # If none exists, retain non-road GIS features rather than returning
    # an empty result.
    # --------------------------------------------------------------------------

    if useful:
        verification_candidates = useful
    else:
        verification_candidates = other_nonroad

    if not verification_candidates:
        return []

    # --------------------------------------------------------------------------
    # DISTANCE-FIRST SORTING
    #
    # Primary:
    #       smallest distance
    #
    # Secondary:
    #       highest criticality
    #
    # Tertiary:
    #       original ordering
    #
    # This is intentionally NOT priority-score-first.
    # --------------------------------------------------------------------------

    indexed = list(enumerate(verification_candidates))

    indexed.sort(
        key=lambda pair: (
            distance(pair[1]),
            -criticality(pair[1]),
            pair[0]
        )
    )

    ranked = []

    for rank, (_, item) in enumerate(indexed, start=1):

        # Preserve the original dictionary completely.
        candidate = dict(item)

        # Update rank only if the field already exists or is naturally
        # expected by the UI.
        candidate["rank"] = rank

        ranked.append(candidate)

    return ranked

