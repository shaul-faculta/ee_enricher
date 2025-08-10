#!/usr/bin/env python3
"""
Setup script for EE Enricher
Helps configure the application with proper environment variables
"""

import os
import sys
import json
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("           EE Enricher Setup")
    print("=" * 60)
    print("Google Earth Engine Data Enrichment Tool")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'earthengine-api', 
        'google-auth', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages...")
        os.system(f"pip install {' '.join(missing_packages)}")
        return True
    return True

def create_env_file():
    """Create .env file from template"""
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if env_file.exists():
        response = input("\n.env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping .env file creation")
            return
    
    if not env_example.exists():
        print("❌ env.example file not found")
        return
    
    print("\n📝 Creating .env file...")
    with open(env_example, 'r') as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ .env file created")
    print("⚠️  Please edit .env file with your actual credentials")

def validate_service_account():
    """Validate service account configuration"""
    print("\n🔍 Validating service account configuration...")
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("❌ .env file not found. Run setup first.")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required variables
    required_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT_ID',
        'SERVICE_ACCOUNT_EMAIL'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var} not set")
        else:
            print(f"✅ {var} = {value}")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please edit .env file with your actual values")
        return False
    
    # Check if credentials file exists
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not Path(creds_path).exists():
        print(f"❌ Credentials file not found: {creds_path}")
        return False
    
    # Validate JSON format
    try:
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        print("✅ Credentials file is valid JSON")
    except json.JSONDecodeError:
        print("❌ Credentials file is not valid JSON")
        return False
    
    return True

def test_earth_engine():
    """Test Earth Engine initialization"""
    print("\n🌍 Testing Earth Engine connection...")
    
    try:
        import ee
        from dotenv import load_dotenv
        load_dotenv()
        
        credentials = ee.ServiceAccountCredentials(
            os.getenv('SERVICE_ACCOUNT_EMAIL'),
            key_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        ee.Initialize(credentials, project=os.getenv('GOOGLE_CLOUD_PROJECT_ID'))
        
        # Test a simple operation
        test_image = ee.Image('USGS/SRTMGL1_003')
        test_point = ee.Geometry.Point([0, 0])
        test_result = test_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=test_point,
            scale=1000
        )
        
        print("✅ Earth Engine connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Earth Engine connection failed: {str(e)}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['uploads', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/ directory")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    check_dependencies()
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Validate configuration
    if validate_service_account():
        if test_earth_engine():
            print("\n🎉 Setup completed successfully!")
            print("\nTo run the application:")
            print("  python app.py")
            print("\nThen open http://localhost:5000 in your browser")
        else:
            print("\n⚠️  Setup completed with warnings")
            print("Earth Engine connection failed - check your credentials")
    else:
        print("\n❌ Setup incomplete")
        print("Please configure your environment variables and run setup again")

if __name__ == "__main__":
    main()
