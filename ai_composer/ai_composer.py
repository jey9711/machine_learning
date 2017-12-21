import urllib
import zipfile
import nottingham_util
import rnn

# url for data collection
midis_url = "www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip"
urllib.urlretrieve(midis_url, "dataset.zip")

# zipping the midis together
zipped = zipfile.ZipFile(r'dataset.zip')
zipped.extractall('data')

# build and train the model
nottingham_util.create_model()
rnn.train_model()