import pickle
import numpy
import numpy as np
from matplotlib.cm import viridis
from sklearn.decomposition import PCA, TruncatedSVD
from PIL import Image
from scipy import sparse
from skimage import transform
import imageio


def get_ds_paths(gs_path):
#     first_n=100
    ds_to_exlude = ['Row001', 'HK_S2_N2_20um_New', '12_cylces_75um_new_submission',
                    '50um_min(focusing)_high(m_z)_dry(application)', '100um_noM2_001_Recal', '50%meoh_8cyc_75um',
                    'Servier_Ctrl_mouse_wb_lateral_plane_DHB', 'North Carolina State University__mouse body',
                    '75um_small(focusing)_high(m_z)_dry(application)', 'Servier_Ctrl_rat_liver_9aa',
                    'Servier_Ctrl_mouse_wb_median_plane_9aa', '20171110_94T_RDAM_1b',
                    '20170905_CGL0170817_MT-MB_ATP_N_81x101_135x135', 'servier_TT_mouse_wb_fmpts_derivatization_CHCA',
                    'Servier_Ctrl_mouse_wb_median_plane_DHB', 'slide077_animal121_rat_kidney - total ion count',
                    '170321_kangaroobrain-bpyn1-pos_maxof50.0_med1r', '70%meoh_8cyc_75um',
                    '75um_small(focusing)_low(m_z)_dry(application)', 'slide012_animal102_minipig_kidney- total ion count']
    all_paths = [p for p in list(gs_path.iterdir()) if p.name not in ds_to_exlude]
    return all_paths


def load_mask(ds_path):
    mask = pickle.load(open(ds_path / 'mask.pkl', 'rb'))
    return mask


def create_ion_list(ds_paths):
    ions = set()
    ds_paths = ds_paths if type(ds_paths) == list else [ds_paths]
    for ds_path in ds_paths:
        for label in ['on', 'off']:
            for img_path in list((ds_path / label).iterdir()):
                ions.add(img_path.name.split('.')[0])
    ions = np.array(sorted(list(ions)))
    return ions


def find_ion_ind(ions, ion):
    t = np.where(np.array(ions) == ion)[0]
    return t[0] if t.shape[0] > 0 else -1


def find_ind(arr, el):
    return find_ion_ind(arr, el)


def create_ds_cube(ds_path, image_shape, ions):
    image_cube = np.zeros(image_shape + ions.shape)
    for label in ['on', 'off']:
        for img_path in list((ds_path / label).iterdir()):
            img = np.array(Image.open(img_path))[:,:,0] / 255.
            ion = img_path.name.split('.')[0]
            ion_ind = find_ion_ind(ions, ion)
            if ion_ind > 0:
                image_cube[:,:,ion_ind] = img
    return image_cube


def random_sample_inds(n, ratio=0.1):
    rand_inds = np.arange(n)
    np.random.shuffle(rand_inds)
    m = int(n * ratio)
    return rand_inds[:m]


def ds_xygroups(ds_path, mask, ions, group_id):
    ds_cube = create_ds_cube(ds_path, mask.shape, ions)
    X = ds_cube.reshape(-1, ds_cube.shape[-1])
    y = mask.reshape(-1)
    groups = np.array([group_id] * y.shape[0])
    # rand_inds = random_sample_inds(X.shape[0], ratio=sample_ratio)
    # return sparse.csr_matrix(X[rand_inds]), y[rand_inds], groups[rand_inds]
    return sparse.csr_matrix(X), y, groups


def load_pixel_xygroups(ds_paths, masks, all_ions):
    X_list, y_list, groups_list = [], [], []
    for i, ds_path in enumerate(ds_paths):
        X, y, groups = ds_xygroups(ds_path, masks[i], all_ions, group_id=i)
        X_list.append(X)
        y_list.append(y)
        groups_list.append(groups)
    return sparse.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)


def load_img_X_y_groups(paths, image_shape=None):
    X_list, y_list, groups = [], [], []
    for i, ds_path in enumerate(paths):
        for img_class in ['on', 'off']:
            for img_path in (ds_path / img_class).iterdir():
                img = imageio.imread(img_path)[:,:,0]
                if image_shape:
                    img = resize_image(img, image_shape)
                img = np.stack([img], axis=2)
                X_list.append(img)
                y_list.append(1 if img_class == 'off' else 0)
                groups.append(i)

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.int32)
    # y = y.reshape(y.shape + (1,)).astype(np.int32)
#     y = keras.utils.to_categorical(y, 2)
    groups = np.stack(groups)
    return X, y, groups


def pca_transform(X_train, X_valid, n_components=100):
    # pca = PCA(n_components=n_components)
    pca = TruncatedSVD(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    return X_train_pca, X_valid_pca


def prepare_mask_match_y(ds_path, ions):
    y = []
    off_ions = [p.name.split('.')[0] for p in (ds_path / 'off').iterdir()]
    on_ions = [p.name.split('.')[0] for p in (ds_path / 'on').iterdir()]
    for ion in ions:
        if ion in off_ions:
            y.append(1)
        elif ion in on_ions:
            y.append(0)
        else:
            y.append(None)
    return np.array(y)


def prepare_mask_match_X(ds_cube, masks):
    X = np.zeros((ds_cube.shape[-1], len(masks)))
    for i in range(ds_cube.shape[-1]):
        img = ds_cube[:,:,i] / 255.
        for j, mask in enumerate(masks):
            corr = np.corrcoef(img.flatten(), mask.flatten())[0,1]
            X[i,j] = corr
    return X


def prepare_mask_match_xygroups(ds_paths, pred_masks):
    X, y, groups = [], [], []
    for ds_path in ds_paths:
        print(ds_path)
        ds_ions = create_ion_list(ds_path)
        masks = pred_masks[ds_path.name]
        masks = masks if type(masks) == list else [masks]
        ds_cube = create_ds_cube(ds_path, masks[0].shape, ds_ions)
        ds_X = prepare_mask_match_X(ds_cube, masks)
        ds_y = prepare_mask_match_y(ds_path, ds_ions)
        ds_groups = [ds_path.name] * ds_y.shape[0]
        X.append(ds_X), y.append(ds_y), groups.append(ds_groups)
    X = np.concatenate(X)
    y = np.concatenate(y)
    groups = np.concatenate(groups)
    return X, y, groups


def gray_to_rgb(img):
    # return viridis(img.squeeze())[:,:,:3]
    img = img.squeeze()
    return np.stack([img, img, img], axis=2)


def convert_to_rgb(X):
    if X.shape[-1] == 3:
        return X

    (img_n, rows, cols) = X.shape[:3]
    X_rgb = np.zeros((img_n, rows, cols, 3))
    for i in range(img_n):
        X_rgb[i] = gray_to_rgb(X[i])
    return X_rgb


def make_subset(u_groups, X, y, groups, to_rgb=False):
    mask = np.array([g in u_groups for g in groups])
    X_sub = X[mask]
    # X_sub.setflags(write=False)
    y_sub = y[mask]
    # y_sub.setflags(write=False)
    groups_sub = groups[mask]
    # groups_sub.setflags(write=False)
    if to_rgb:
        X_sub = convert_to_rgb(X_sub)
    return X_sub, y_sub, groups_sub


def resize_image(img, image_shape=(64, 64)):
    img = transform.resize(img, image_shape, mode='constant')
    img = img / img.max()
    return img


def read_resize(img_path):
    img = imageio.imread(img_path)[:,:,0]
    img = resize_image(img)
    return img
