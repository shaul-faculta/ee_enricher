# EE Enricher

A web application for enriching CSV datasets containing soil sampling locations with geospatial variables from Google Earth Engine (GEE).

## Features

- **CSV Upload & Column Mapping**: Upload CSV files and automatically detect/map columns for point ID, latitude, and longitude
- **Earth Engine Layer Selection**: Choose from curated Earth Engine datasets including TerraClimate, MODIS NDVI, WorldClim, and SRTM elevation data
- **Batch Data Processing**: Efficient batch processing with progress tracking and error handling
- **Service Account Authentication**: Secure authentication using Google Cloud service accounts
- **Modern Web Interface**: Beautiful, responsive web interface with drag-and-drop file upload
- **Data Preview & Download**: Preview enriched data and download results as CSV

## Prerequisites

Before setting up EE Enricher, you need:

1. **Google Cloud Project** with Earth Engine API enabled
2. **Service Account** with appropriate Earth Engine permissions
3. **Python 3.8+** installed on your system

## Setup Instructions

### 1. Google Cloud & Earth Engine Setup

#### Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Earth Engine API for your project

#### Register for Earth Engine Access
1. Visit [Earth Engine Signup](https://signup.earthengine.google.com/)
2. Sign up with your Google account
3. Wait for approval (usually takes 24-48 hours)

#### Create a Service Account
1. In Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Give it a name (e.g., "ee-enricher-service")
4. Grant the following roles:
   - Earth Engine Resource Viewer
   - Service Usage Consumer
5. Create and download the JSON key file
6. **Important**: Register the service account for Earth Engine access

### 2. Application Setup

#### Clone the Repository
```bash
git clone <repository-url>
cd ee_enricher
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Configure Environment Variables
1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` with your actual values:
   ```bash
   # Path to your service account JSON key file
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
   
   # Your Google Cloud Project ID
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   
   # Service account email
   SERVICE_ACCOUNT_EMAIL=your-service-account@your-project-id.iam.gserviceaccount.com
   ```

#### Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment (free)

### Why not Vercel?
Vercel’s free tier is optimized for serverless/edge functions and short-lived requests. This app needs a long-running Python process and persistent state for Earth Engine calls, which doesn’t fit Vercel’s model. Use Render’s free tier or Fly.io instead.

### Deploy on Render (free)
1. Push this repo to GitHub.
2. On Render, create a new Web Service from your repo. It will detect `render.yaml` (Docker) and build.
3. Add environment variables in the Render dashboard:
   - `GOOGLE_APPLICATION_CREDENTIALS`: path to your service account JSON in the container (e.g., `/opt/render/project/src/service-account.json`)
   - `GOOGLE_CLOUD_PROJECT_ID`: your project ID
   - `SERVICE_ACCOUNT_EMAIL`: your service account email
4. Provide the service account JSON via Render “Secret Files” and set `GOOGLE_APPLICATION_CREDENTIALS` accordingly.
5. Deploy; after build completes, Render gives you a public URL to share with your team.

## Usage

### 1. Upload CSV File
- Drag and drop a CSV file or click to browse
- The file should contain columns for point ID, latitude, and longitude
- Supported formats: CSV files up to 16MB

### 2. Map Columns
- The application will automatically suggest column mappings
- Confirm or modify the mappings for:
  - Point ID column
  - Latitude column
  - Longitude column

### 3. Select Earth Engine Layer
Choose from available datasets:
- **TerraClimate**: Global climate and water balance data (monthly, 4km)
- **MODIS NDVI**: Vegetation index data (16-day, 250m)
- **WorldClim**: Global climate data (annual, 30s)
- **SRTM**: Digital elevation data (static, 30m)

### 4. Set Date Range (Optional)
- For temporal datasets, specify start and end dates
- Leave empty for static datasets

### 5. Enrich Data
- Click "Start Enrichment" to begin processing
- Monitor progress in real-time
- Download the enriched CSV when complete

## API Endpoints

- `GET /` - Main application interface
- `POST /api/upload` - Upload CSV file
- `GET /api/layers` - Get available Earth Engine layers
- `POST /api/enrich` - Enrich data with Earth Engine variables
- `GET /api/download/<filename>` - Download enriched CSV file
- `GET /api/health` - Health check endpoint

## Supported Earth Engine Datasets

### TerraClimate
- **Variables**: Temperature (min/max), precipitation, evapotranspiration, etc.
- **Resolution**: 4km
- **Temporal**: Monthly (1958-present)

### MODIS NDVI
- **Variables**: Normalized Difference Vegetation Index
- **Resolution**: 250m
- **Temporal**: 16-day composites

### WorldClim
- **Variables**: 19 bioclimatic variables
- **Resolution**: 30 arc-seconds
- **Temporal**: Annual averages

### SRTM
- **Variables**: Elevation, slope, aspect
- **Resolution**: 30m
- **Temporal**: Static

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Missing or incorrect column mappings
- Earth Engine authentication issues
- Quota exceeded errors
- Network timeouts

## Security Considerations

- Service account credentials are stored securely via environment variables
- No user authentication required - all processing uses the configured service account
- File uploads are validated and sanitized
- Temporary files are cleaned up automatically

## Troubleshooting

### Common Issues

1. **"Earth Engine not initialized"**
   - Check that your service account JSON key file path is correct
   - Verify the service account has Earth Engine access
   - Ensure the Google Cloud project ID is correct

2. **"Quota exceeded" errors**
   - The application automatically handles batching and retries
   - For large datasets, consider processing in smaller batches

3. **"Invalid coordinates" error**
   - Ensure latitude values are between -90 and 90
   - Ensure longitude values are between -180 and 180

### Getting Help

- Check the system status indicator on the web interface
- Review the application logs for detailed error messages
- Verify your Earth Engine access and service account permissions

## Development

### Project Structure
```
ee_enricher/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main interface
├── uploads/           # Uploaded and processed files
├── env.example        # Environment variables template
└── README.md          # This file
```

### Adding New Earth Engine Layers

To add new datasets, modify the `get_available_layers()` function in `app.py` and add corresponding processing logic in `enrich_data_with_gee()`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Google Earth Engine team for providing the platform
- Flask community for the web framework
- Bootstrap and Font Awesome for the UI components
