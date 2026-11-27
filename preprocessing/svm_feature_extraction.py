import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split

DATA_DIR = Path("dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CACHE_DIR = Path("features_cache")

CLASSES = {"REAL": 0, "FAKE": 1}
CLASS_NAMES = ["REAL", "FAKE"]

IMG_SIZE = 32

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

LBP_P, LBP_R = 8, 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_P + 2
LBP_GRID = 4

RANDOM_STATE = 42
N_JOBS = -1


def load_image_paths(split_dir):
    paths, labels = [], []
    for class_name, label in CLASSES.items():
        files = sorted((Path(split_dir) / class_name).glob("*.*"))
        paths.extend(str(f) for f in files)
        labels.extend([label] * len(files))
    return np.array(paths), np.array(labels)


def load_gray(path):
    img = Image.open(path).convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)


def extract_hog(gray, visualize=False):
    return hog(gray, feature_vector=True, visualize=visualize, **HOG_PARAMS)


def extract_lbp(gray, return_map=False):
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    step = IMG_SIZE // LBP_GRID
    hists = []
    for i in range(0, IMG_SIZE, step):
        for j in range(0, IMG_SIZE, step):
            region = lbp[i:i + step, j:j + step]
            h, _ = np.histogram(region, bins=LBP_BINS, range=(0, LBP_BINS))
            hists.append(h / region.size)
    feats = np.concatenate(hists)
    return (feats, lbp) if return_map else feats


def extract_features(path):
    gray = load_gray(path)
    return np.concatenate([extract_hog(gray), extract_lbp(gray)]).astype(np.float32)


def feature_lengths():
    gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    return extract_hog(gray).shape[0], extract_lbp(gray).shape[0]


def build_feature_matrix(paths, labels, cache_name, cache_dir=CACHE_DIR, n_jobs=N_JOBS):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        if np.array_equal(data["y"], labels):
            print(f"Loaded cached features: {cache_file}")
            return data["X"], data["y"]
        print("Cache does not match dataset - recomputing.")

    t0 = time.time()
    feats = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(extract_features)(p) for p in tqdm(paths, desc=f"Extracting {cache_name}")
    )
    X = np.vstack(feats)
    np.savez_compressed(cache_file, X=X, y=labels)
    print(f"Extracted {X.shape} in {time.time() - t0:.1f}s -> cached to {cache_file}")
    return X, labels


def stratified_subset(X, y, n, seed=RANDOM_STATE):
    if n is None or n >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_sub, y_sub


def load_dataset_features():
    train_paths, y_train = load_image_paths(TRAIN_DIR)
    test_paths, y_test = load_image_paths(TEST_DIR)
    X_train, y_train = build_feature_matrix(train_paths, y_train, "train_hog_lbp")
    X_test, y_test = build_feature_matrix(test_paths, y_test, "test_hog_lbp")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    n_hog, n_lbp = feature_lengths()
    print(f"HOG: {n_hog}  LBP: {n_lbp}  Combined: {n_hog + n_lbp}")
    X_train, y_train, X_test, y_test = load_dataset_features()
    print("X_train:", X_train.shape, " X_test:", X_test.shape)
