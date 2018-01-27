import os
from urllib.request import urlopen
from zipfile import ZipFile

def make_dataset(args):
    
	image_urls = ['https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip',
		         'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip',
		         'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip']

	annotation_urls = ['https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip',
		                'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip',
		                'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip']


	image_destination = os.path.join(args.raw_image_directory, 'images')
	annotation_destination = os.path.join(args.raw_image_directory, 'annotations')
	if not os.path.isdir(image_destination):
		print('Downloading and unzipping datasets...')
		os.mkdir(image_destination)
		os.mkdir(annotation_destination)

		download_unzip_files(image_urls, image_destination)
		download_unzip_files(annotation_urls, annotation_destination)
	else:
		print('Image files are already saved in image directory')  

def download_unzip_files(url_list, destination):
	for url in url_list:
		zip_resp = urlopen(url)
		temp_zip = open('/tmp/tempfile.zip', 'wb')
		temp_zip.write(zip_resp.read())
		temp_zip.close()
		zf = ZipFile('/tmp/tempfile.zip')
		zf.extractall(path=destination)
		zf.close


