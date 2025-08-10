import ee
print("ee version:", ee.__version__)
credentials = ee.ServiceAccountCredentials(
    'geoenrich-app-runner@soil-minerals.iam.gserviceaccount.com',
    key_file='C:/Code/ee_enricher/service-account-key-d5c42905d29b.json'
)
print("Credentials created successfully!")
