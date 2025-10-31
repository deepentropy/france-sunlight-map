from glob import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rio_cogeo import cog_translate, cog_profiles
from rasterio.windows import Window
from tqdm import tqdm
import logging, sys
import os, traceback
from pathlib import Path
import tempfile

# Choix techniques
COG_COMPRESS = "ZSTD"      # "ZSTD" (si GDAL>=3.5) sinon "DEFLATE"
COG_LEVEL = 15             # ZSTD level
BLOCKSIZE = 512            # tuilage interne 512x512
OUTPUT_DTYPE = "int16"     # "int16" recommandé, sinon "float32"
INT16_NODATA = -32768      # standard int16
FLOAT_NODATA = -99999.0    # si float32
FORCE_CRS = "EPSG:2154"    # Lambert-93

MAX_WORKERS = min(max(2, os.cpu_count()//2), 12)  # ajuste si NVMe rapide
GDAL_THREADS_PER_PROC = "1"                        # **clé** pour éviter l’oversubscription
GDAL_CACHE_MB = "256"

def convert_task(src_path: str, dst_path: str, dtype: str):
    # Contexte par processus
    os.environ["GDAL_NUM_THREADS"] = GDAL_THREADS_PER_PROC
    os.environ["GDAL_CACHEMAX"] = GDAL_CACHE_MB
    try:
        info = convert_one(src_path, str(dst_path), dtype=dtype)  # ta fonction existante corrigée
        return {"ok": True, "src": src_path, "dst": str(dst_path), "detail": info}
    except Exception as e:
        return {"ok": False, "src": src_path, "dst": str(dst_path),
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def setup_logger(name="asc2cog", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger

log = setup_logger()

def find_asc_files(root: str, pattern: str):
    paths = glob(os.path.join(root, pattern), recursive=True)
    paths = [p for p in paths if p.lower().endswith(".asc")]
    log.info(f"{len(paths)} fichiers .asc trouvés")
    return paths

def derive_output_path(src_path: str, out_dir: str):
    p = Path(src_path)
    # Extrait l'ID département si présent dans le dossier IGN
    # ...RGEALTI_MNT_5M_ASC_LAMB93_IGN69_D001\RGEALTI_FXX_0830_6540_MNT_LAMB93_IGN69.asc
    dept = "DXXX"
    for part in p.parts:
        if part.startswith("RGEALTI_MNT_5M_ASC_LAMB93_IGN69_"):
            dept = part.split("_")[-1]  # D001
            break
    # Extrait tuile "0830_6540"
    tile = p.stem.replace("RGEALTI_FXX_", "").replace("_MNT_LAMB93_IGN69", "")
    dst = Path(out_dir) / f"{dept}_{tile}_MNT5m.tif"
    return dst

def read_src_meta(src_path: str):
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        meta = {
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "nodata": prof.get("nodata", None),
            "cellsize": src.transform.a,  # 5.0 m attendu
        }
    log.debug(f"Meta {Path(src_path).name}: {meta}")
    return meta


def _cog_profile():
    prof = cog_profiles.get("deflate")
    prof.update({
        "blockxsize": BLOCKSIZE,
        "blockysize": BLOCKSIZE,
        "overview_resampling": Resampling.nearest.name,
        "bigti": "IF_SAFER",
    })
    if COG_COMPRESS.upper() == "ZSTD":
        prof["compress"] = "ZSTD"
        prof["zstd_level"] = COG_LEVEL
    else:
        prof["compress"] = "DEFLATE"
        prof["predictor"] = 2
    return prof

def _write_intermediate_gtiff(src_path: str, tmp_tif: str, dtype: str, dst_nodata):
    meta = read_src_meta(src_path)

    # Profil d’écriture rasterio
    dst_profile = {
        "driver": "GTiff",
        "width": meta["width"],
        "height": meta["height"],
        "count": 1,
        "crs": FORCE_CRS,                 # force EPSG:2154
        "transform": meta["transform"],
        "dtype": dtype,
        "nodata": dst_nodata,
        "tiled": True,
        "blockxsize": BLOCKSIZE,
        "blockysize": BLOCKSIZE,
        "compress": "DEFLATE",            # léger pour l’intermédiaire
        "predictor": 2
    }

    with rasterio.open(src_path) as src, rasterio.open(tmp_tif, "w", **dst_profile) as dst:
        # Écriture par blocs pour faible RAM
        tile = 1024
        for y in range(0, meta["height"], tile):
            h = min(tile, meta["height"] - y)
            for x in range(0, meta["width"], tile):
                w = min(tile, meta["width"] - x)
                win = Window(x, y, w, h)
                arr = src.read(1, window=win, masked=False)

                if dtype == "int16":
                    src_nodata = meta["nodata"]
                    if src_nodata is not None:
                        arr = np.where(arr == src_nodata, INT16_NODATA, arr)
                    arr = np.clip(arr, -32768, 32767).astype(np.int16, copy=False)
                else:
                    if meta["nodata"] is None:
                        # rien à remapper, on reste en float32
                        pass
                    arr = arr.astype(np.float32, copy=False)

                dst.write(arr, 1, window=win)

    return tmp_tif

def convert_one(src_path: str, dst_path: str, dtype=OUTPUT_DTYPE):
    meta = read_src_meta(src_path)

    # NODATA cible
    if dtype == "int16":
        dst_nodata = INT16_NODATA
    else:
        dst_nodata = FLOAT_NODATA if meta["nodata"] is None else float(meta["nodata"])

    # Fichier intermédiaire
    tmp_dir = Path(tempfile.gettempdir())
    tmp_tif = str(tmp_dir / (Path(dst_path).stem + ".tmp.tif"))

    #log.info(f"[stage1] ASC→GTiff intermédiaire: {src_path} → {tmp_tif}")
    _write_intermediate_gtiff(src_path, tmp_tif, dtype, dst_nodata)

    # COG final
    cog_prof = _cog_profile()
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    #log.info(f"[stage2] GTiff→COG: {tmp_tif} → {dst_path}")
    try:
        cog_translate(
            tmp_tif,                      # chemin source
            str(dst_path),
            cog_prof,
            in_memory=False,
            quiet=True,
            nodata=dst_nodata,
            dtype=dtype,
        )
    except TypeError as e:
        # Fallback si options non supportées
        log.warning(f"Retry cog_translate sans dtype/nodata ({e})")
        cog_translate(tmp_tif, str(dst_path), cog_prof, in_memory=False, quiet=True)

    # Nettoyage
    try:
        os.remove(tmp_tif)
    except Exception as e:
        log.warning(f"Temp non supprimé: {tmp_tif} ({e})")

    return {"dst": dst_path, "dtype": dtype, "nodata": dst_nodata}

def batch_convert(asc_list, out_dir: str, dtype=OUTPUT_DTYPE, overwrite=False, dryrun=False):
    results = []
    for src in tqdm(asc_list, desc="ASC→COG"):
        dst = derive_output_path(src, out_dir)
        if not overwrite and Path(dst).exists():
            log.info(f"Skip (existe): {dst}")
            results.append({"dst": str(dst), "skipped": True})
            continue
        if dryrun:
            log.info(f"Dryrun: {src} -> {dst}")
            results.append({"dst": str(dst), "dryrun": True})
            continue
        try:
            info = convert_one(src, str(dst), dtype=dtype)
            results.append(info)
        except Exception as e:
            log.error(f"Erreur: {src} -> {e}")
    return results

def validate_cog(path: str):
    with rasterio.open(path) as ds:
        ok = (ds.driver == "GTiff"
              and ds.block_shapes is not None
              and ds.profile.get("compress", None) is not None
              and ds.count == 1)
        meta = {
            "size": (ds.width, ds.height),
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs),
            "nodata": ds.nodatavals[0],
            "block": ds.block_shapes[0],
            "compress": ds.profile.get("compress", None),
        }
    if ok:
        log.info(f"COG OK: {path} | {meta}")
    else:
        log.warning(f"COG douteux: {path} | {meta}")
    return ok, meta

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def parallel_batch_convert(asc_list, out_dir: str, dtype="int16", overwrite=False):
    jobs = []
    for src in asc_list:
        dst = derive_output_path(src, out_dir)
        if not overwrite and Path(dst).exists():
            log.info(f"Skip (existe): {dst}")
            continue
        jobs.append((src, dst, dtype))

    if not jobs:
        log.info("Rien à convertir.")
        return []

    results = []
    log.info(f"Lancement parallélisé: {len(jobs)} fichiers, {MAX_WORKERS} workers, "
             f"{GDAL_THREADS_PER_PROC} thread GDAL/proc, cache {GDAL_CACHE_MB} Mo/proc")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(convert_task, src, dst, dtype) for src, dst, dtype in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="ASC→COG parallel"):
            r = f.result()
            if r["ok"]:
                log.debug(f"OK: {r['dst']}")
            else:
                log.error(f"FAIL: {r['src']} -> {r['error']}")
            results.append(r)
    return results
