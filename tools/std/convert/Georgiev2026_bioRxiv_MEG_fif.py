"""Minimal FIF (Neuromag/MNE) reader for Georgiev2026_bioRxiv_MEG: reads the tag stream of a raw .fif file with plain
struct/numpy (the FIF tag format is public: 16-byte big-endian tag header kind/type/size/next, then the data) and returns
ONLY the non-MEG channels it is asked for (stimulus, EMG, MISC, EOG, ECG). MEG and EEG channels are never copied out of the
file buffers. Nothing in the file is executed."""
import struct
import numpy as np

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
