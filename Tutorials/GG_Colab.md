# This tutorial for Google Colab


### Check whether TPU is enabled

```
import os
import pprint
import tensorflow as tf
if ‘COLAB_TPU_ADDR’ not in os.environ:
	print(‘ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!’)
else:
	tpu_address = ‘grpc://’ + os.environ[‘COLAB_TPU_ADDR’]
	print (‘TPU address is’, tpu_address)
	with tf.Session(tpu_address) as session:
		devices = session.list_devices()
		print(‘TPU devices:’)
		pprint.pprint(devices)
 ```
 
 ### Install libraries

Colab supports both the `pip` and `apt` package managers.

```
!pip install torch
!apt install -y graphviz
```

### Upload Datasets

#### Code to upload from Local

```
from google.colab import files
uploaded = files.upload()
```

#### Upload files from Google Drive

You can get id of the file you want to upload,and use the above code.

For more resource to upload files from [google services](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=bRFyEsdfBxJ9).

```
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
# PyDrive reference:
# https://gsuitedevs.github.io/PyDrive/docs/build/html/index.html
# 2. Create & upload a file text file.
uploaded = drive.CreateFile({'title': 'Sample upload.txt'})
uploaded.SetContentString('Sample upload file content')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))
# 3. Load a file by ID and print its contents.
downloaded = drive.CreateFile({'id': uploaded.get('id')})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))
```
