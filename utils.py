import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from requests import get
import pandas as pd

image_api_endpoint = 'https://metaspace2020.eu'


def fetch_save_data_as_png(img_url, f_path):
    print(img_url, f_path)

    if not f_path.exists():
        if not f_path.parent.exists():
            f_path.parent.mkdir(parents=True, exist_ok=True)

        data = get(img_url).content
        img = Image.open(io.BytesIO(data)).convert('LA')
        img.save(f_path)


def save_ion_images(uri_list, image_path_list):
    arg_list = [(image_api_endpoint + uri, path)
                for (uri, path) in zip(uri_list, image_path_list)]

    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda args: fetch_save_data_as_png(*args), arg_list))


def download_tagged_images(data_path, prefix, tagger_df):

    def ion_image_path(row):
        if row.type == 2:
            label = 'off'
        elif row.type == 1:
            label = 'on'
        else:
            label = 'other'

        path = (Path(prefix)
                / row.dsName.replace('/', '_')
                / label
                / f'{row.sumFormula}{row.adduct}.png')
        return path

    uri_list = [row.ionImageUrl for row in tagger_df.itertuples()]
    image_path_list = [ion_image_path(row) for row in tagger_df.itertuples()]
    label_list = ['off' if row.type == 2 else 'on' for row in tagger_df.itertuples()]

    save_ion_images(uri_list, [data_path / p for p in image_path_list])

    path_label_df = pd.DataFrame(zip(image_path_list, label_list), columns=['path', 'label'])
    path_label_df.to_csv(data_path / 'path_label.csv', index=False)
