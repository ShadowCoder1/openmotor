"""Georgiev2026_bioRxiv_MEG: resumable batch job (run on the PC with pc.py py). For each MEG .fif of the movement-execution
tasks (OSF 9fgw8, about 430-500 MB each) it downloads the file, checks the OSF md5, reads ONLY the EMG, MISC (force,
accelerometer, audio cue) and STIM channels with the minimal FIF reader, stores them in a small compressed npz in
tmp_samples/Georgiev2026_bioRxiv_MEG/ and deletes the raw file. MEG channels are never copied out of the file. One file on
disk at a time; a log line per file goes to tmp_samples/Georgiev2026_bioRxiv_MEG/log.txt."""
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

DS = "Georgiev2026_bioRxiv_MEG"
ROOT = Path.cwd()
RAW = ROOT / "raw_extra" / DS
TMP = ROOT / "tmp_samples" / DS
FILES = [["MEG3720/DHmovement1_tsss_mc_ica.fif", "https://osf.io/download/68fb6c182911c5c06c7f2cde/", 433624607, "72710654c1f2b3531f60894ec08ea169"], ["MEG3720/IHI1_tsss_mc_ica.fif", "https://osf.io/download/68fb6c1e52dd8e6c2af97cf9/", 436328639, "69e60a294af93b64abd73ea6603dee13"], ["MEG3735/DHmovement1_tsss_mc_ica.fif", "https://osf.io/download/68fb6cf5931a498af6f98ab9/", 430916575, "ae6820d368c657651901ac5ab3bc0a28"], ["MEG3735/IHI1_tsss_mc_ica.fif", "https://osf.io/download/68fb6d07f815be4dbbc4010d/", 444436739, "ddbef087fe45c35f7680c49abd85098e"], ["MEG3746/DHmovement1_tsss_mc_ica.fif", "https://osf.io/download/68fb6d93f226dd1a1d4970ea/", 434972731, "76365e950fc31c106527e2e52a1dee14"], ["MEG3746/IHI1_tsss_mc_ica.fif", "https://osf.io/download/68fb6d9aed1cca91621a007a/", 430916683, "dac348fa639b4f3cf28377cccfff2fd1"], ["MEG3773/IHI1_tsss_mc_ica.fif", "https://osf.io/download/68fb6e5bed1cca91621a00fe/", 433620787, "0c639d99ac3c42e4f668906a94802086"], ["MEG3773/DHmovement_tsss_mc_ica.fif", "https://osf.io/download/68fb6e5e9ff2ab78db19ffa8/", 429564739, "ccd35f41614579551d0a1c3ec5292284"], ["MEG3783/DHmovement_tsss_mc_ica.fif", "https://osf.io/download/68fb6ecb9ff2ab78db19fff8/", 429564559, "d7b510769bafa3816dfacfef7668702c"], ["MEG3783/IHI1_tsss_mc_ica.fif", "https://osf.io/download/68fb6ee2732fa344a47f2fad/", 437676659, "7d60da63552bb55ae19ca2a8622c1f6d"], ["MEG4124/IHI1_tsss_mc_ica.fif", "https://osf.io/download/68fb6f36f226dd1a1d49729c/", 436382276, "4144371ced079b4ac1469af7062c670a"], ["MEG4124/DHmovement2_tsss_mc_ica.fif", "https://osf.io/download/68fb6f3f5fa3a101387f2c18/", 433678244, "6e700a8b90ccfd3994f53b7c674e0db6"], ["MEG4125/DHmovement1_tsss_mc_ica.fif", "https://osf.io/download/69b92a66687f6fa83fe38139/", 433663160, "8dc52d1124165fc32c0db1685ad6f897"], ["MEG4125/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92a67b3de3f278ce38125/", 433663160, "b591d61e6b8c2c1f88289006974a2a1a"], ["MEG4126/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b929e8caf519ec049f001f/", 436368164, "34df7371028b0e9a7c4deeabff05680d"], ["MEG4126/DHmovement2_tsss_mc_ica.fif", "https://osf.io/download/69b929e9527ba320a79f0045/", 449888324, "896be04553c689c3f53f882e8f1fa2a9"], ["MEG4127/DHmovement1_tsss_mc_ica.fif", "https://osf.io/download/69b92b7ffbd8238c16a73018/", 435028888, "41263493da159f9eb538e44c08e7dbe5"], ["MEG4127/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92b83fbd8238c16a73036/", 451253080, "e552616bb1f233b591a6ee2dee0a314b"], ["MEG4131/DHmovement3_tsss_mc_ica.fif", "https://osf.io/download/69b92cca4af368992b9f0325/", 444492568, "3c9361dffc9c7ccc3942b371d7f17924"], ["MEG4131/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92cccfbd8238c16a73253/", 460716760, "47840ee9c5a49c6aef2afcf28f2924a7"], ["MEG5130/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92d41caf519ec049f0345/", 441765136, "6300234515a994a19be57be6ed3e1bc3"], ["MEG5130/DHmovement2_tsss_mc_ica.fif", "https://osf.io/download/69b92d414af368992b9f0379/", 441765136, "21b5450e2fcccf346666e99c27aaadd5"], ["MEG5131/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92dcfc5c9dade29a72cb2/", 432356720, "9400b1cb9d11789a86c66c63e67e7c7d"], ["MEG5131/DHmovement2_tsss_mc_ica.fif", "https://osf.io/download/69b92dd1e92f3753ec9f0748/", 432356720, "de04e5e1395f699aa7a5d9ff56bae458"], ["MEG5132/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92e97fbd8238c16a7343b/", 431004596, "cd9d1ff41833c849868dda78733c9dec"], ["MEG5132/DHmovement2_tsss_mc_ica.fif", "https://osf.io/download/69b92e98d737d18438e3826f/", 435060644, "b73989b7eb5753298200944c3c97b0d3"], ["MEG5133/IHI2_tsss_mc_ica.fif", "https://osf.io/download/69b92f31687f6fa83fe386d4/", 426949588, "24b5ab207df834a64355df824e88ec8e"], ["MEG5133/DHmovement2_tsss_mc_ica.fif", "https://osf.io/download/69b92f31140e23047ea73420/", 431005640, "c764a0241599a640c3a4d41ebd56fc7d"], ["MEG5388/meg_5388_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269091cc447a4d7e6d96e7/", 501205070, "ac24c37901bb1353f46440edea110936"], ["MEG5396/meg_5396_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269114695f202b8a867282/", 441886818, "eb4a5d389b80028d8afcbde46378a343"], ["MEG5397/meg_5397_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a26917a466519c9d2867292/", 454023750, "0b337ce7db68e75b6875b8aff814563e"], ["MEG5404/meg_5404_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a26921a07980523968672a9/", 459404366, "ef662f9c24a1850d0961a2b837ccee25"], ["MEG5408/meg_5408_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269278ec49ba16d097fa62/", 455385734, "3dec8f5cc42c71277b8cec1dfec8105a"], ["MEG5413/meg_5413_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a2692d2479530fcc497fc99/", 451312742, "dde31608bbfb5a3d9aebe6947db2db82"], ["MEG5414/meg_5414_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269345ecee4a80c197fc40/", 466113778, "76fb95e8e6af24e25d379eff2c1dc671"], ["MEG5415/meg_5415_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269780fc832dec4997fa54/", 443224466, "4823a76768f84c63589c806d2298a6c9"], ["MEG5421/meg_5421_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a2697fd4b2e20cba56d8275/", 462107742, "df01b6c1b66cb81fd2409e1f8f0423d9"], ["MEG5423/meg_5423_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a26985352e16db6a986738f/", 460747022, "8dba53ce447765843bd165eb950d18d4"], ["MEG5435/meg_5435_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a2698bf3e7c9ca90d97fc6b/", 471525062, "b2a4b20f8cc0467aabac914ac89a5fa2"], ["MEG5444/meg_5444_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a2699223e7c9ca90d97fc9c/", 440544778, "9afc59737e791b44ae8784a4cc4eebf7"], ["MEG5450/meg_5450_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a26997e3e69ce4b1097fcb2/", 474237834, "7b34f7f683530ea049a646b07e59f8f4"], ["MEG5454/meg_5454_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a2699de808bcea8906d8285/", 475598666, "03a551c3188abd2074efda555f9d8d98"], ["MEG5474/meg_5474_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269a4052e16db6a98674cb/", 448631974, "9c5e788d32ff5120946fc6cf5e35c7a9"], ["MEG5475/meg_5475_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269aa74b2e20cba56d8472/", 443224034, "d2f637555d667f627d1cbade1407dcae"], ["MEG5480/meg_5480_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269aff5adaea5f8b8672e5/", 456718922, "32e1074645d1969a86fb2c908c2d694c"], ["MEG5491/meg_5491_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269b682f5101f6e86d8255/", 454003122, "41f09414d6da1b935c7798a6fc58b88e"], ["MEG5496/meg_5496_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269bc125690c59cf6d8252/", 454027962, "a88c35c9c1d16236c6b9f040e84500dd"], ["MEG5501/meg_5501_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269c3100acd23335867423/", 443234154, "acd847e1a11dc572df1687c251f3aa69"], ["MEG5542/meg_5542_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269c91c9ad3c4d6a8672d1/", 449965450, "fa7bfd3ffe71e422abee4f1083ce3974"], ["MEG5543/meg_5543_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269d55c9ad3c4d6a867482/", 439200614, "3c2a92807889e9a561f2ead725583651"], ["MEG5553/meg_5553_t5_2_tsss_mc_ica.fif", "https://osf.io/download/6a269dbef73414f3f36d8223/", 451325202, "b05e0cd91d4021265bf9de6aaa608de1"], ["MEG5570/meg_5570_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269e0f2f5101f6e86d83d8/", 441898122, "7316e5eb5e32099b2895b39b0e74dbb2"], ["MEG5574/meg_5574_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269e6a5adaea5f8b86a205/", 448628914, "7242a6512b11a1ee03ea11c3ff41ead8"], ["MEG5577/meg_5577_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269ebdd4f72b7311867337/", 444577810, "ce5b263968367b2697b0eefab473b511"], ["MEG5583/meg_5583_t5_1_tsss_mc_ica.fif", "https://osf.io/download/6a269f2fc4f4d6c71b97fb58/", 455386782, "4dd8bf9b40e663c1739eff25e17e7f2e"]]
KEEP_KINDS = ("EMG", "MISC", "STIM")


# ---- minimal FIF reader (same code as Georgiev2026_bioRxiv_MEG_fif.py)
import struct

FIFF_BLOCK_START, FIFF_BLOCK_END = 104, 105
FIFF_NCHAN, FIFF_SFREQ, FIFF_CH_INFO, FIFF_DATA_BUFFER = 200, 201, 203, 300
FIFF_FIRST_SAMPLE, FIFF_DATA_SKIP, FIFF_DATA_SKIP_SAMP = 208, 304, 305
FIFFB_RAW_DATA = 102
TYPES = {2: ">i2", 3: ">i4", 4: ">f4", 5: ">f8", 16: ">i2"}
KIND_NAMES = {1: "MEG", 2: "EEG", 3: "STIM", 202: "EOG", 302: "EMG", 402: "ECG", 502: "MISC", 201: "REF_MEG", 301: "SYST"}


def _chinfo(b):
    scanno, logno, kind = struct.unpack(">3i", b[:12])
    rng, cal = struct.unpack(">2f", b[12:20])
    coil = struct.unpack(">i", b[20:24])[0]
    unit, mul = struct.unpack(">2i", b[72:80])
    name = b[80:96].split(b"\0")[0].decode("ascii", "replace")
    return dict(scanno=scanno, logno=logno, kind=kind, kind_name=KIND_NAMES.get(kind, str(kind)), range=rng, cal=cal,
                unit=unit, unit_mul=mul, name=name)


def read_info(path, want=None):
    """Return dict(sfreq, nchan, chs, first_samp, buffers: list of (offset, type, nsamp_per_chan)).  Scans every tag once."""
    info = dict(chs=[], buffers=[], first_samp=0, sfreq=None, nchan=None, skips=[])
    with open(path, "rb") as fh:
        depth_raw = 0
        pos = 0
        blocks = []
        while True:
            h = fh.read(16)
            if len(h) < 16:
                break
            kind, typ, size, nxt = struct.unpack(">4i", h)
            dpos = fh.tell()
            if kind == FIFF_BLOCK_START:
                blocks.append(struct.unpack(">i", fh.read(4))[0])
            elif kind == FIFF_BLOCK_END:
                if blocks:
                    blocks.pop()
            elif kind == FIFF_NCHAN and blocks and blocks[-1] == 101 and info["nchan"] is None:
                info["nchan"] = struct.unpack(">i", fh.read(4))[0]
            elif kind == FIFF_SFREQ and blocks and blocks[-1] == 101 and info["sfreq"] is None:
                info["sfreq"] = struct.unpack(">f", fh.read(4))[0]
            elif kind == FIFF_CH_INFO and blocks and blocks[-1] == 101 and len(info["chs"]) < (info["nchan"] or 10 ** 9):
                info["chs"].append(_chinfo(fh.read(size)))
            elif kind == FIFF_FIRST_SAMPLE and not info["buffers"]:
                info["first_samp"] = struct.unpack(">i", fh.read(4))[0]
            elif kind == FIFF_DATA_SKIP:
                info["skips"].append((len(info["buffers"]), struct.unpack(">i", fh.read(4))[0]))
            elif kind == FIFF_DATA_BUFFER:
                info["buffers"].append((dpos, typ, size))
            if nxt > 0:
                fh.seek(nxt)
            elif nxt == 0:
                fh.seek(dpos + size)
            else:
                break
    return info


def read_channels(path, info, names):
    """Samples of the named channels as float64 arrays in the file's units times cal*range (SI as MNE does)."""
    idx = {c["name"]: i for i, c in enumerate(info["chs"])}
    nch = len(info["chs"])
    rows = [idx[n] for n in names]
    out = [[] for _ in names]
    with open(path, "rb") as fh:
        for dpos, typ, size in info["buffers"]:
            dt = np.dtype(TYPES[typ])
            ns = size // dt.itemsize // nch
            fh.seek(dpos)
            buf = np.frombuffer(fh.read(size), dtype=dt).reshape(ns, nch).T
            for k, r in enumerate(rows):
                c = info["chs"][r]
                out[k].append(buf[r].astype(np.float64) * c["cal"] * c["range"])
    return {n: np.concatenate(o) for n, o in zip(names, out)}


def fetch(url, dest, size, md5):
    for attempt in range(5):
        try:
            h = hashlib.md5()
            with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as fh:
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    fh.write(b)
                    h.update(b)
            assert dest.stat().st_size == size, (dest.stat().st_size, size)
            assert h.hexdigest() == md5, "md5 differs from OSF"
            return
        except Exception as e:  # retry
            last = e
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {last}")


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 6000.0
    t_start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    log = open(TMP / "log.txt", "a")
    for rel, url, size, md5 in FILES:
        folder, name = rel.split("/")
        outp = TMP / f"{folder}__{name}.npz"
        if outp.exists():
            continue
        if time.time() - t_start > budget:
            print("time budget used; rerun to continue")
            return
        free = shutil.disk_usage(ROOT).free / 1e9
        if free < 8:
            print(f"free {free:.1f} GB < 8, stopping")
            return
        path = RAW / f"{folder}__{name}"
        t = time.time()
        fetch(url, path, size, md5)
        info = read_info(str(path))
        keep = [c["name"] for c in info["chs"] if c["kind_name"] in KEEP_KINDS]
        d = read_channels(str(path), info, keep)
        arr = np.stack([d[k] for k in keep]).astype(np.float32)
        np.savez_compressed(outp, names=np.array(keep), data=arr, sfreq=info["sfreq"], first_samp=info["first_samp"],
                            n_meg=sum(c["kind_name"] == "MEG" for c in info["chs"]), nchan=info["nchan"])
        path.unlink()
        sti = d.get("STI101")
        ev = "" if sti is None else str(np.unique(sti, return_counts=True))[:200].replace("\n", " ")
        q = {k: [round(float(x), 4) for x in np.percentile(d[k], [0, 1, 50, 99, 100])] for k in keep}
        msg = f"{folder}/{name} {size/1e6:.0f} MB ok in {time.time()-t:.0f}s n={arr.shape[1]} sti={ev} pct(0,1,50,99,100)={q}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    print("all files done")


if __name__ == "__main__":
    main()
