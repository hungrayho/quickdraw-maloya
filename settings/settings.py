import os

PATH = os.getcwd()
BUCKET_NAME = 'gs://quickdraw_dataset'
SOURCE_BLOB_NAME = '/full/numpy_bitmap/dog.npy'
DESTINATION_FILE_NAME = PATH + os.sep + 'data'