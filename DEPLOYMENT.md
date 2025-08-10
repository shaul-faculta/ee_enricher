# EE Enricher - Deployment Guide

This guide provides step-by-step instructions for deploying the EE Enricher application.

## Quick Start

### 1. Prerequisites Check
```bash
# Check Python version (3.8+ required)
python --version

# Check if pip is available
pip --version
```

### 2. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd ee_enricher

# Run the setup script
python setup.py
```

### 3. Configure Credentials
1. Edit the `.env` file with your actual values:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   SERVICE_ACCOUNT_EMAIL=your-service-account@your-project-id.iam.gserviceaccount.com
   ```

### 4. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Detailed Setup Instructions

### Google Cloud & Earth Engine Setup

#### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your Project ID (you'll need this later)

#### Step 2: Enable Earth Engine API
1. In Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Earth Engine API"
3. Click on it and press "Enable"

#### Step 3: Register for Earth Engine Access
1. Visit [Earth Engine Signup](https://signup.earthengine.google.com/)
2. Sign up with your Google account
3. Wait for approval (usually 24-48 hours)
4. Once approved, you'll receive an email confirmation

#### Step 4: Create Service Account
1. In Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Name: `ee-enricher-service`
4. Description: `Service account for EE Enricher application`
5. Click "Create and Continue"

#### Step 5: Grant Permissions
1. Add the following roles:
   - **Earth Engine Resource Viewer**
   - **Service Usage Consumer**
2. Click "Continue"
3. Click "Done"

#### Step 6: Create and Download Key
1. Click on your service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose "JSON" format
5. Download the key file
6. **Important**: Store this file securely and never commit it to version control

#### Step 7: Register Service Account for Earth Engine
1. Go to [Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Sign in with your Google account
3. In the Code Editor, run this script:
   ```javascript
   // Register service account for Earth Engine
   var serviceAccountEmail = 'your-service-account@your-project-id.iam.gserviceaccount.com';
   ee.data.authenticateViaPrivateKey(serviceAccountEmail, function() {
     print('Service account registered successfully');
   });
   ```

### Application Deployment

#### Option 1: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

#### Option 2: Production Deployment (Gunicorn)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Option 3: Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
# Build and run
docker build -t ee-enricher .
docker run -p 5000:5000 --env-file .env ee-enricher
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON key file | Yes |
| `GOOGLE_CLOUD_PROJECT_ID` | Your Google Cloud Project ID | Yes |
| `SERVICE_ACCOUNT_EMAIL` | Service account email address | Yes |
| `FLASK_ENV` | Flask environment (development/production) | No |
| `FLASK_DEBUG` | Enable Flask debug mode | No |
| `UPLOAD_FOLDER` | Custom upload folder path | No |

## Security Considerations

### Production Deployment
1. **Use HTTPS**: Always use HTTPS in production
2. **Secure Credentials**: Store service account keys securely
3. **Environment Variables**: Use environment variables, not hardcoded values
4. **File Upload Limits**: Configure appropriate file size limits
5. **Access Control**: Implement user authentication if needed

### Credential Management
- Never commit service account keys to version control
- Use secret management services in production
- Rotate credentials regularly
- Use least-privilege access

## Monitoring and Logging

### Application Logs
The application logs to stdout/stderr. In production, configure proper logging:

```python
# Add to app.py for production logging
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/ee_enricher.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('EE Enricher startup')
```

### Health Checks
The application provides a health check endpoint:
```bash
curl http://localhost:5000/api/health
```

## Troubleshooting

### Common Issues

1. **"Earth Engine not initialized"**
   - Check service account JSON file path
   - Verify service account has Earth Engine access
   - Ensure project ID is correct

2. **"Quota exceeded" errors**
   - Check Earth Engine quotas in Google Cloud Console
   - Implement request batching for large datasets
   - Consider upgrading your Earth Engine quota

3. **"Invalid credentials" error**
   - Verify service account key file is valid JSON
   - Check service account email matches key file
   - Ensure service account is registered for Earth Engine

4. **"File upload failed"**
   - Check file size limits (default: 16MB)
   - Verify file is valid CSV format
   - Check upload directory permissions

### Debug Mode
Enable debug mode for detailed error messages:
```bash
export FLASK_DEBUG=True
python app.py
```

### Testing
Use the provided sample data to test the application:
```bash
# Upload sample_data.csv through the web interface
# This file contains 10 sample soil sampling locations
```

## Performance Optimization

### For Large Datasets
1. **Batch Processing**: The application processes data in batches
2. **Caching**: Implement caching for frequently accessed data
3. **Async Processing**: Consider async processing for large files
4. **Database**: Use a database for large datasets instead of CSV files

### Scaling Considerations
1. **Load Balancing**: Use multiple application instances
2. **Database**: Implement proper database for production use
3. **Caching**: Add Redis or similar for caching
4. **CDN**: Use CDN for static assets

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review application logs
3. Verify Earth Engine access and quotas
4. Check Google Cloud Console for service account status

## Updates and Maintenance

### Regular Maintenance
1. **Update Dependencies**: Regularly update Python packages
2. **Monitor Logs**: Check application logs for errors
3. **Backup Data**: Backup uploaded and processed files
4. **Security Updates**: Keep system and dependencies updated

### Version Updates
1. Check for new versions of the application
2. Review changelog for breaking changes
3. Test updates in development environment
4. Deploy updates during maintenance windows
