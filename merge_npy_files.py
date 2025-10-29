# python
import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

def _iter_arrays_from_file(path: str):
    """Retourne (arr, name) pour chaque tableau présent dans `path` (.npy ou .npz)."""
    loaded = np.load(path, mmap_mode='r')
    # .npz -> mapping, .npy -> ndarray or memmap
    if hasattr(loaded, 'files'):
        for key in loaded.files:
            yield loaded[key], f"{os.path.basename(path)}:{key}"
    else:
        yield loaded, os.path.basename(path)

def merge_npy_files(input_dir: str, output_path: str, axis: int = 0, chunk_bytes: int = 50 * 1024 * 1024, dtype: Optional[np.dtype] = None):
    """
    Concatène tous les tableaux trouvés dans `input_dir` (*.npy, *.npz) le long de `axis`
    et écrit le résultat sur `output_path` (ex : `D:\\rgealti\\combined.npy`).
    Limite la mémoire en lisant/écrivant par blocs (par défaut ~50MiB).
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")) + glob.glob(os.path.join(input_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .npy/.npz dans {input_dir}")

    # 1) Collecter formes et dtype
    print("Collecte des formes et types de données...")
    shapes: List[Tuple[int, ...]] = []
    dtypes = []
    for f in tqdm(files):
        for arr, name in _iter_arrays_from_file(f):
            if arr.ndim == 0:
                raise ValueError(f"Fichier {f} contient un scalaire (attendu 1D+).")
            shapes.append(arr.shape)
            dtypes.append(arr.dtype)

    # Vérifier compatibilité pour concaténation
    print("Vérification des formes pour concaténation...")
    ref_shape = list(shapes[0])
    for sh in tqdm(shapes[1:]):
        for i, (a, b) in enumerate(zip(ref_shape, sh)):
            if i == axis:
                continue
            if a != b:
                raise ValueError("Dimensions non compatibles pour concaténation (axes autres que axis doivent correspondre).")

    # Calculer forme finale
    total_axis = sum(s[axis] for s in shapes)
    out_shape = list(ref_shape)
    out_shape[axis] = total_axis
    out_shape = tuple(out_shape)

    # Choisir dtype de sortie
    out_dtype = np.dtype(dtype) if dtype is not None else np.result_type(*dtypes)

    # Créer memmap de sortie (mode 'w+' crée/écrase)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode='w+', dtype=out_dtype, shape=out_shape)

    # 2) Copier fichier par fichier en blocs
    print("Copie des données...")
    write_index = 0
    for f in tqdm(files, desc="Fichiers"):
        for arr, name in _iter_arrays_from_file(f):
            length = arr.shape[axis]
            # taille d'une "ligne" selon axis pour calculer rows_per_chunk
            bytes_per_element = arr.dtype.itemsize if arr.dtype is not None else out_dtype.itemsize
            # nombre d'éléments par "row" = product des dims hors axis
            elems_per_row = int(np.prod([s for i, s in enumerate(arr.shape) if i != axis]))
            bytes_per_row = elems_per_row * bytes_per_element
            # si bytes_per_row == 0 (protection), écrire tout d'un coup
            if bytes_per_row == 0:
                out_slice = [slice(None)] * out.ndim
                out_slice[axis] = slice(write_index, write_index + length)
                out[tuple(out_slice)] = arr.astype(out_dtype, copy=False)
                write_index += length
                continue

            rows_per_chunk = max(1, int(chunk_bytes // bytes_per_row))
            # copier par tranches sur l'axe
            for start in range(0, length, rows_per_chunk):
                end = min(length, start + rows_per_chunk)
                # construire slices
                src_slice = [slice(None)] * arr.ndim
                src_slice[axis] = slice(start, end)
                dst_slice = [slice(None)] * out.ndim
                dst_slice[axis] = slice(write_index + start, write_index + end)
                block = arr[tuple(src_slice)]
                if block.dtype != out_dtype:
                    block = block.astype(out_dtype, copy=False)
                out[tuple(dst_slice)] = block
            write_index += length

    # flush explicitement
    del out
    return output_path

# Exemple d'utilisation :
# merged = merge_npy_files(input_dir=`npy`, output_path=`D:\\rgealti\\combined.npy`)
# print("Fichier écrit :", merged)
