# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=100):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')


# %%
# !conda list

# %%
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('sfpd police car unit', max_images=1)
urls[0]

# %%
from fastdownload import download_url
dest = 'sfpd.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)

# %%
download_url(search_images('car sedan photo', max_images=1)[0], 'car.jpg', show_progress=False)
Image.open('car.jpg').to_thumb(256,256)

# %%
searches = 'car sedan','sfpd police car unit'
path = Path('sfpd_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# %%
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# %%
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=12)

# %%
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(8)

# %%
is_sfpd,what,probs = learn.predict(PILImage.create('sfpd.jpg'))
print(f"This is a: {is_sfpd}.")
print(f"Probability it's SFPD: {probs[1]:.4f}")

# %%
# prediction with a photo that is not a car at all

is_sfpd,what,probs = learn.predict(PILImage.create('jhoward.jpeg'))
print(f"This is a: {is_sfpd}.")
print(f"Probability it's SFPD: {probs[1]:.4f}")

# %%
# let's see how the network behaves when classifying photos of non cars in general.
# Begin by downloading a few hundred photos of whatever street.
path = Path('any_street')

dest = path
dest.mkdir(exist_ok=True, parents=True)
download_images(dest, urls=search_images(f'street photo'))
sleep(10)  # Pause between searches to avoid over-loading server
download_images(dest, urls=search_images(f'street sun photo'))
sleep(10)
download_images(dest, urls=search_images(f'street shade photo'))
sleep(10)
resize_images(path, max_size=400, dest=path)

# %%
dls_street = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls_street.show_batch(max_n=3)

# %%
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# %%
cops = 0
other = 0

cop_files = []

for f in get_image_files(path):
    i = PILImage.create(f)
    result = learn.predict(i)
    if result[1] == 1:
        cops += 1
        cop_files.append(result + (f,))
    else:
        other += 1

print(f"Totals: {cops} cops and {other} other.")

# %%
for r in cop_files:
    print(r)

# %%
results
