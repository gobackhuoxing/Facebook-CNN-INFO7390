import requests

#Because the facebook picture url is in a special format, Google API can not use it directly.
#So, we will download the picture and prcess it then delete it.

def download_picture(uri,path):
    with open(path, 'wb') as handle:
            response = requests.get(uri, stream=True)

            if not response.ok:
                print (response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
				
#path = 'temp/tempPic.jpg'
#uri = 'https://external.xx.fbcdn.net/safe_image.php?d=AQDkwsYg7VqP2ZJv&w=130&h=130&url=http%3A%2F%2Fcdn.akamai.steamstatic.com%2Fsteam%2Fapps%2F711660%2Fheader.jpg%3Ft%3D1520261021&cfs=1&_nc_hash=AQDAzFptZ2Afori2'

#download_picture(uri,path)