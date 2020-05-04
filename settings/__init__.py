from settings import settings
import os

if not os.path.exists(settings.DESTINATION_FILE_NAME):
    os.mkdir(settings.DESTINATION_FILE_NAME)