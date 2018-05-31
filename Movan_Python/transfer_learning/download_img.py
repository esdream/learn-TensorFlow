import os
from urllib.request import urlretrieve
def download():

    categories = ['tiger', 'kittycat']

    for category in categories:
        os.makedirs('./img/%s' % category, exist_ok=True)
        with open('./img_url/imagenet_%s.txt' % category, 'r') as url_file:
            urls = url_file.readlines()
            n_urls = len(urls)
            for i, url in enumerate(urls):
                try:
                    print(url)
                    urlretrieve(url.strip(), './img/%s%s' % (category, url.strip().split('/')[-1]))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no image')

if(__name__ == '__main__'):
    download()
