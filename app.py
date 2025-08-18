import os
import re
import json
import math
import pandas as pd
import ee
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import logging
from dotenv import load_dotenv
import threading
import time
import uuid
from typing import Optional, List, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory job store. For production, replace with Redis or a database.
jobs = {}

# Helpers for Earth Engine asset handling
def detect_dataset_type_and_first_image(dataset_id: str) -> tuple[str, ee.Image]:
    """Return (type, first_image). Type is 'Image' or 'ImageCollection'.
    For ImageCollections returns the first image (unfiltered)."""
    # Try as Image first
    try:
        image = ee.Image(dataset_id)
        # Force a small server-side call to validate
        _ = image.bandNames().size().getInfo()
        return 'Image', image
    except Exception:
        # Fallback to ImageCollection
        collection = ee.ImageCollection(dataset_id)
        image = collection.first()
        # Validate the first image
        _ = image.bandNames().size().getInfo()
        return 'ImageCollection', image

def build_image_from_asset(dataset_id: str, selected_variables, date_range: Optional[dict]) -> ee.Image:
    """Build an ee.Image from an asset ID. If it's a collection, filter by date_range and take first.
    selected_variables can be a string or list of strings."""
    try:
        # Try as single Image
        image = ee.Image(dataset_id).select(selected_variables)
        # Trigger validation
        _ = image.bandNames().size().getInfo()
        return image
    except Exception:
        # Treat as ImageCollection
        collection = ee.ImageCollection(dataset_id)
        if date_range and date_range.get('start') and date_range.get('end'):
            collection = collection.filterDate(date_range['start'], date_range['end'])
        image = collection.select(selected_variables).first()
        # Validate
        _ = image.bandNames().size().getInfo()
        return image

# Initialize Earth Engine
def initialize_earth_engine():
    """Initialize Earth Engine with service account credentials"""
    try:
        # Check for service account credentials
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
        
        if credentials_path and project_id:
            credentials = ee.ServiceAccountCredentials(
                os.getenv('SERVICE_ACCOUNT_EMAIL'),
                key_file=credentials_path
            )
            ee.Initialize(credentials, project=project_id)
            logger.info("Earth Engine initialized with service account")
            return True
        else:
            logger.error("Missing Earth Engine credentials in environment variables")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {str(e)}")
        return False

# Initialize Earth Engine on startup
# Prefer environment-provided credentials. Fall back to local file only if present (dev only).
if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    local_sa = './service-account-key-d5c42905d29b.json'
    if os.path.exists(local_sa):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_sa

ee_initialized = initialize_earth_engine()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_csv_columns(csv_file):
    """Detect and suggest column mappings for pointid, lat, lon"""
    try:
        df = pd.read_csv(csv_file, nrows=5)  # Read first 5 rows for preview
        columns = df.columns.tolist()
        
        # Auto-detect common column names
        suggestions = {
            'pointid': None,
            'lat': None,
            'lon': None
        }
        
        # Look for common variations
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['id', 'point', 'sample', 'site']):
                suggestions['pointid'] = col
            elif any(keyword in col_lower for keyword in ['lat', 'latitude', 'y']):
                suggestions['lat'] = col
            elif any(keyword in col_lower for keyword in ['lon', 'long', 'longitude', 'x']):
                suggestions['lon'] = col
        
        # Build JSON-safe preview (replace NaN/NA with None)
        preview_df = df.copy()
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        return {
            'columns': columns,
            'preview': preview_df.to_dict('records'),
            'suggestions': suggestions
        }
    except Exception as e:
        logger.error(f"Error detecting CSV columns: {str(e)}")
        return None

def get_available_layers():
    """Get a curated list of available Earth Engine layers"""
    layers = [
        {
            'id': 'TERRACLIMATE',
            'name': 'TerraClimate',
            'description': 'Global climate and water balance data',
            'gee_id': 'IDAHO_EPSCOR/TERRACLIMATE',
            'variables': ['tmmn', 'tmmx', 'ppt', 'pet', 'aet', 'def', 'pdsi', 'vpd', 'vap', 'srad', 'ws', 'th', 'pdsi', 'ro', 'swe', 'soil'],
            'temporal_resolution': 'monthly',
            'spatial_resolution': '4km'
        },
        {
            'id': 'MODIS_NDVI',
            'name': 'MODIS NDVI',
            'description': 'Normalized Difference Vegetation Index (MODIS/061/MOD13Q1)',
            'gee_id': 'MODIS/061/MOD13Q1',
            'variables': ['NDVI'],
            'temporal_resolution': '16-day',
            'spatial_resolution': '250m'
        },
        {
            'id': 'WORLDCLIM',
            'name': 'WorldClim',
            'description': 'Global climate data',
            'gee_id': 'WORLDCLIM/V1/BIO',
            'variables': ['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 'bio07', 'bio08', 'bio09', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19'],
            'temporal_resolution': 'annual',
            'spatial_resolution': '30s'
        },
        {
            'id': 'SRTM',
            'name': 'SRTM Digital Elevation',
            'description': 'Digital elevation data',
            'gee_id': 'USGS/SRTMGL1_003',
            'variables': ['elevation'],
            'temporal_resolution': 'static',
            'spatial_resolution': '30m'
        },
        {
            'id': 'CUSTOM',
            'name': 'Custom Dataset',
            'description': 'Enter a custom GEE Image or ImageCollection ID',
            'gee_id': None,
            'variables': [],
            'temporal_resolution': 'variable',
            'spatial_resolution': 'variable'
        }
    ]
    return layers

def _chunk_dataframe_rows(df: pd.DataFrame, batch_size: int):
    """Yield row chunks of the DataFrame with size up to batch_size."""
    num_rows = len(df)
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        yield df.iloc[start:end]

def _make_json_safe(value: Any) -> Any:
    """Recursively replace NaN/Inf values with None so JSON is valid."""
    try:
        # Use pandas to detect NA for scalars
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if value is None:
            return None
        # Pandas NA-like
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, dict):
            return {k: _make_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_make_json_safe(v) for v in value]
        return value
    except Exception:
        return value

def _safe_enrich_with_retries(points_df: pd.DataFrame, layer_config: dict, date_range: Optional[dict], max_retries: int = 3, base_delay: float = 1.0) -> pd.DataFrame:
    """Call enrich_data_with_gee with retries and exponential backoff."""
    attempt = 0
    while True:
        try:
            return enrich_data_with_gee(points_df, layer_config, date_range)
        except Exception as exc:  # noqa: BLE001 broad except to implement retries around EE errors
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_seconds = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Batch enrichment failed (attempt {attempt}/{max_retries}). Retrying in {sleep_seconds:.1f}s. Error: {exc}")
            time.sleep(sleep_seconds)

def _process_enrichment_job(job_id: str, filepath: str, column_mapping: dict, layer_config: dict, date_range: Optional[dict], batch_size: int, accumulate: bool = False, base_output_filename: Optional[str] = None):
    """Background job to enrich data in batches and update job status."""
    try:
        jobs[job_id].update({
            'status': 'running',
            'message': 'Reading input CSV'
        })
        jobs[job_id]['start_time'] = time.time()
        jobs[job_id]['avg_batch_seconds'] = None

        # Determine base DataFrame: either original upload or previously enriched output (for accumulation)
        base_df_path = None
        if accumulate and base_output_filename:
            candidate_path = os.path.join(app.config['UPLOAD_FOLDER'], base_output_filename)
            if os.path.exists(candidate_path):
                base_df_path = candidate_path
        original_df = pd.read_csv(base_df_path or filepath)

        # Rename columns according to mapping
        working_df = original_df.rename(columns={
            column_mapping['pointid']: 'pointid',
            column_mapping['lat']: 'lat',
            column_mapping['lon']: 'lon'
        })

        # Clean up any duplicate-suffixed columns from prior merges (e.g., *_x, *_y)
        def _collapse_duplicate_band_columns(df: pd.DataFrame) -> pd.DataFrame:
            suffix_regex = re.compile(r'^(?P<base>.+)_(x|y)$')
            base_to_suffix_cols = {}
            for col in list(df.columns):
                match = suffix_regex.match(col)
                if match and match.group('base') != 'pointid':
                    base = match.group('base')
                    base_to_suffix_cols.setdefault(base, []).append(col)
            # Combine suffix columns into base and drop suffix versions
            for base, cols in base_to_suffix_cols.items():
                if base in df.columns:
                    series = df[base]
                    for c in cols:
                        series = series.combine_first(df[c])
                    df[base] = series
                else:
                    series = None
                    for c in cols:
                        series = df[c] if series is None else series.combine_first(df[c])
                    df[base] = series
                df.drop(columns=cols, inplace=True, errors='ignore')
            return df

        working_df = _collapse_duplicate_band_columns(working_df)

        # Validate coordinates
        if not all(working_df['lat'].between(-90, 90)) or not all(working_df['lon'].between(-180, 180)):
            jobs[job_id].update({
                'status': 'failed',
                'message': 'Invalid coordinates detected in input data'
            })
            return

        total_rows = len(working_df)
        total_batches = (total_rows + batch_size - 1) // batch_size
        jobs[job_id].update({
            'total_batches': total_batches,
            'completed_batches': 0,
            'message': f'Starting enrichment in {total_batches} batches'
        })

        enriched_parts: List[pd.DataFrame] = []

        for batch_index, chunk_df in enumerate(_chunk_dataframe_rows(working_df[['pointid', 'lat', 'lon']], batch_size), start=1):
            # Support cancellation
            if jobs.get(job_id, {}).get('canceled'):
                jobs[job_id].update({
                    'status': 'canceled',
                    'message': 'Job canceled by user'
                })
                return

            jobs[job_id].update({'message': f'Processing batch {batch_index}/{total_batches}'})

            batch_start = time.time()
            batch_enriched = _safe_enrich_with_retries(chunk_df, layer_config, date_range)
            enriched_parts.append(batch_enriched)
            batch_duration = time.time() - batch_start

            # Update progress
            jobs[job_id]['completed_batches'] = batch_index
            progress = int((batch_index / total_batches) * 100)
            jobs[job_id]['progress'] = progress
            # Update running average and ETA
            prev_avg = jobs[job_id].get('avg_batch_seconds')
            if prev_avg is None:
                jobs[job_id]['avg_batch_seconds'] = batch_duration
            else:
                jobs[job_id]['avg_batch_seconds'] = (prev_avg * (batch_index - 1) + batch_duration) / batch_index
            remaining = max(total_batches - batch_index, 0)
            jobs[job_id]['eta_seconds'] = int(remaining * jobs[job_id]['avg_batch_seconds'])

        # Merge all enriched results
        if enriched_parts:
            all_enriched = pd.concat(enriched_parts, ignore_index=True)
        else:
            all_enriched = pd.DataFrame(columns=['pointid'])

        # When accumulating, avoid duplicate columns by dropping incoming columns that already exist
        incoming_df = all_enriched
        if accumulate:
            existing_cols = set(working_df.columns)
            safe_cols = ['pointid'] + [c for c in incoming_df.columns if c != 'pointid' and c not in existing_cols]
            incoming_df = incoming_df[safe_cols]

        result_df = working_df.merge(incoming_df, on='pointid', how='left')
        # Final cleanup of any residual duplicate-suffix columns
        result_df = _collapse_duplicate_band_columns(result_df)

        # Save enriched CSV (reuse same output for accumulation, or create new on first pass)
        output_filename = base_output_filename if (accumulate and base_output_filename) else f"enriched_{os.path.basename(filepath)}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        result_df.to_csv(output_path, index=False)

        # Preview (ensure JSON-safe: convert NaN/NaT to None)
        preview_df = result_df.head(10).copy()
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        preview_records = preview_df.to_dict('records')

        jobs[job_id].update({
            'status': 'completed',
            'message': 'Enrichment completed',
            'output_filename': output_filename,
            'preview': _make_json_safe(preview_records),
            'progress': 100,
            'eta_seconds': 0
        })
    except Exception as exc:  # noqa: BLE001 broad except to ensure job failure is captured
        logger.exception(f"Job {job_id} failed: {exc}")
        jobs[job_id].update({
            'status': 'failed',
            'message': f'Enrichment failed: {str(exc)}',
            'last_error': str(exc)
        })

@app.route('/api/dataset_bands', methods=['POST'])
def get_dataset_bands():
    """Fetch available bands/variables for a given dataset (built-in or custom).
    Auto-detects if the asset is an Image or ImageCollection."""
    try:
        data = request.get_json()
        dataset_id = data.get('gee_id')
        if not dataset_id:
            return jsonify({'error': 'Missing dataset ID'}), 400
        if not ee_initialized:
            return jsonify({'error': 'Earth Engine not initialized'}), 500
        dataset_type, image = detect_dataset_type_and_first_image(dataset_id)
        band_names = image.bandNames().getInfo()
        scale_m = image.projection().nominalScale().getInfo()
        return jsonify({'bands': band_names, 'type': dataset_type, 'scale_m': scale_m})
    except Exception as e:
        logger.error(f"Error fetching dataset bands: {str(e)}")
        return jsonify({'error': f'Failed to fetch bands: {str(e)}'}), 500

def enrich_data_with_gee(points_df, layer_config, date_range=None):
    """Enrich point data with Earth Engine data"""
    try:
        if not ee_initialized:
            raise Exception("Earth Engine not initialized")
        features = []
        for _, row in points_df.iterrows():
            point = ee.Geometry.Point([row['lon'], row['lat']])
            feature = ee.Feature(point, {'pointid': row['pointid']})
            features.append(feature)
        points_fc = ee.FeatureCollection(features)
        # Determine selected variables (support multiple)
        selected_variables = layer_config.get('selected_variables')
        if not selected_variables:
            single_var = layer_config.get('selected_variable')
            selected_variables = [single_var] if single_var else []
        if not selected_variables:
            raise Exception('No variables selected')

        # Temporal strategy options for ImageCollections
        temporal_strategy = layer_config.get('temporal_strategy') or 'mean'  # default to 'mean'
        include_std = bool(layer_config.get('include_std'))

        # Helper to build an aggregated image from an ImageCollection
        def build_image_from_collection(collection: ee.ImageCollection, bands: List[str]) -> ee.Image:
            coll = collection.select(bands)
            # If a date range is provided, filter accordingly
            if date_range and date_range.get('start') and date_range.get('end'):
                coll = coll.filterDate(date_range['start'], date_range['end'])

            # Choose base image based on strategy
            if temporal_strategy == 'mean':
                base_img = coll.mean()
            elif temporal_strategy == 'oldest':
                base_img = coll.sort('system:time_start', True).first()
            else:  # default and 'most_recent'
                base_img = coll.sort('system:time_start', False).first()

            if include_std:
                std_img = coll.reduce(ee.Reducer.stdDev())
                # Ensure std band names end with _std matching selected order
                std_band_names = [f"{b}_std" for b in bands]
                std_img = std_img.rename(std_band_names)
                return base_img.addBands(std_img)
            return base_img

        # Get the appropriate Earth Engine dataset based on layer_config
        if layer_config['id'] == 'TERRACLIMATE':
            dataset = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
            image = build_image_from_collection(dataset, selected_variables)
        elif layer_config['id'] == 'MODIS_NDVI':
            dataset = ee.ImageCollection('MODIS/061/MOD13Q1')
            image = build_image_from_collection(dataset, selected_variables)
        elif layer_config['id'] == 'WORLDCLIM':
            image = ee.ImageCollection('WORLDCLIM/V1/BIO').first().select(selected_variables)
        elif layer_config['id'] == 'SRTM':
            image = ee.Image('USGS/SRTMGL1_003').select(selected_variables)
        elif layer_config['id'] == 'CUSTOM':
            # Custom dataset: if temporal strategy provided, treat as ImageCollection; otherwise use auto-detect builder
            gee_id = layer_config['gee_id']
            if temporal_strategy:
                collection = ee.ImageCollection(gee_id)
                image = build_image_from_collection(collection, selected_variables)
            else:
                # Auto-detect and select first image if ImageCollection
                image = build_image_from_asset(gee_id, selected_variables, date_range)
        else:
            raise Exception(f"Unsupported layer: {layer_config['id']}")
        
        # Apply bilinear resampling for continuous curated layers to improve accuracy
        if layer_config['id'] in {'TERRACLIMATE', 'MODIS_NDVI', 'WORLDCLIM', 'SRTM', 'CUSTOM'}:
            image = image.resample('bilinear')

        # Use the dataset's native nominal scale (derive from first band)
        native_scale = image.projection().nominalScale().getInfo()

        # Extract values at points. Use sampleRegions so band names are preserved as columns
        result = image.sampleRegions(
            collection=points_fc,
            properties=['pointid'],
            scale=native_scale,
            geometries=False
        )
        result_list = result.getInfo()['features']
        enriched_data = []
        for feature in result_list:
            properties = feature['properties']
            enriched_data.append(properties)
        return pd.DataFrame(enriched_data)
    except Exception as e:
        logger.error(f"Error enriching data with GEE: {str(e)}")
        raise

@app.route('/')
def index():
    """Main application page"""
    try:
        layers = get_available_layers()
    except Exception:
        layers = []
    return render_template('index.html', server_layers=layers)

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and column detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect columns
            column_info = detect_csv_columns(filepath)
            if column_info is None:
                return jsonify({'error': 'Failed to read CSV file'}), 400
            
            return jsonify(_make_json_safe({
                'message': 'File uploaded successfully',
                'filename': filename,
                'column_info': column_info
            }))
        else:
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/layers', methods=['GET'])
def get_layers():
    """Get available Earth Engine layers"""
    try:
        layers = get_available_layers()
        return jsonify({'layers': layers})
    except Exception as e:
        logger.error(f"Error getting layers: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/enrich', methods=['POST'])
def enrich_data():
    """Enrich CSV data with Earth Engine data"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        column_mapping = data.get('column_mapping')
        layer_config = data.get('layer_config')
        date_range = data.get('date_range')
        
        if not all([filename, column_mapping, layer_config]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Read the uploaded CSV
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        
        # Rename columns according to mapping
        df = df.rename(columns={
            column_mapping['pointid']: 'pointid',
            column_mapping['lat']: 'lat',
            column_mapping['lon']: 'lon'
        })
        
        # Validate coordinates
        if not all(df['lat'].between(-90, 90)) or not all(df['lon'].between(-180, 180)):
            return jsonify({'error': 'Invalid coordinates detected'}), 400
        
        # Enrich data with GEE
        enriched_df = enrich_data_with_gee(df, layer_config, date_range)
        
        # Merge with original data
        result_df = df.merge(enriched_df, on='pointid', how='left')
        
        # Save enriched data
        output_filename = f"enriched_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        result_df.to_csv(output_path, index=False)
        
        # Build preview with JSON-safe values
        preview_df = result_df.head(10).copy()
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        return jsonify(_make_json_safe({
            'message': 'Data enriched successfully',
            'output_filename': output_filename,
            'preview': preview_df.to_dict('records')
        }))
        
    except Exception as e:
        logger.error(f"Error enriching data: {str(e)}")
        return jsonify({'error': f'Enrichment failed: {str(e)}'}), 500


@app.route('/api/enrich/start', methods=['POST'])
def start_enrichment_job():
    """Start background enrichment with batching; returns a job_id to poll for progress."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        column_mapping = data.get('column_mapping')
        layer_config = data.get('layer_config')
        date_range = data.get('date_range')
        batch_size = int(data.get('batch_size', 500))
        accumulate = bool(data.get('accumulate', False))
        base_output_filename = data.get('base_output_filename')

        if not all([filename, column_mapping, layer_config]):
            return jsonify({'error': 'Missing required parameters'}), 400

        if not ee_initialized:
            return jsonify({'error': 'Earth Engine not initialized'}), 500

        # Validate that at least one variable/band is selected
        if not (layer_config.get('selected_variable') or layer_config.get('selected_variables')):
            return jsonify({'error': 'No variable(s) selected for the chosen layer'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Uploaded file not found'}), 404

        # If the client did not explicitly request accumulation but a prior enriched file exists,
        # default to appending to that file to preserve previously pulled columns.
        try:
            default_output_filename = f"enriched_{os.path.basename(filepath)}"
            default_output_path = os.path.join(app.config['UPLOAD_FOLDER'], default_output_filename)
            if not accumulate and os.path.exists(default_output_path):
                accumulate = True
                base_output_filename = default_output_filename
        except Exception:
            pass

        job_id = uuid.uuid4().hex
        jobs[job_id] = {
            'status': 'pending',
            'progress': 0,
            'message': 'Queued',
            'total_batches': None,
            'completed_batches': 0,
            'eta_seconds': None,
            'canceled': False
        }

        thread = threading.Thread(
            target=_process_enrichment_job,
            args=(job_id, filepath, column_mapping, layer_config, date_range, batch_size, accumulate, base_output_filename),
            daemon=True,
        )
        thread.start()

        return jsonify({'job_id': job_id})
    except Exception as e:
        logger.error(f"Error starting enrichment job: {str(e)}")
        return jsonify({'error': f'Failed to start job: {str(e)}'}), 500


@app.route('/api/enrich/status/<job_id>', methods=['GET'])
def get_enrichment_status(job_id: str):
    """Get current status/progress of a background enrichment job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job.get('status'),
        'progress': job.get('progress', 0),
        'message': job.get('message'),
        'total_batches': job.get('total_batches'),
        'completed_batches': job.get('completed_batches'),
        'eta_seconds': job.get('eta_seconds'),
        'last_error': job.get('last_error')
    })


@app.route('/api/enrich/cancel/<job_id>', methods=['POST'])
def cancel_enrichment_job(job_id: str):
    """Request cancellation of a running enrichment job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.get('status') not in {'running', 'pending'}:
        return jsonify({'error': 'Job is not running'}), 400
    job['canceled'] = True
    job['message'] = 'Cancel requested'
    return jsonify({'status': 'canceling'})


@app.route('/api/enrich/result/<job_id>', methods=['GET'])
def get_enrichment_result(job_id: str):
    """Get the result metadata (preview and output filename) when job completes."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.get('status') != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    return jsonify(_make_json_safe({
        'message': 'Data enriched successfully',
        'output_filename': job.get('output_filename'),
        'preview': job.get('preview', [])
    }))

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download enriched CSV file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'earth_engine_initialized': ee_initialized,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, host='0.0.0.0', port=port)
