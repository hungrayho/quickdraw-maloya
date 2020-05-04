from settings import settings
from downloader.gcs import download_blob

if __name__ == "__main__":
    download_blob(settings.BUCKET_NAME, settings.SOURCE_BLOB_NAME, settings.DESTINATION_FILE_NAME)
