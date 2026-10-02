"""Fake competition root for testing the submission notebook locally: tiny synthetic DICOM series, never real data.

Usage: python tests/make_fake_dicoms.py <root> [n_studies]
Creates test.csv, test_series.csv, sample_submission.csv, train.csv and test_series/<study>/<series>/<n>.dcm.
One study lacks its axial series and the last study in sample_submission.csv has no files at all (fallback paths).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = {  # plane: (ImageOrientationPatient, axis along which slices are stacked)
    "Sagittal": ([0, 1, 0, 0, 0, -1], 0),
    "Coronal": ([1, 0, 0, 0, 0, -1], 1),
    "Axial": ([1, 0, 0, 0, 1, 0], 2),
}


def write_series(folder, plane, rng, n_slices=10, size=64):
    folder.mkdir(parents=True, exist_ok=True)
    iop, axis = PLANES[plane]
    for k in range(n_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.Manufacturer, ds.MagneticFieldStrength = "MR", "FAKE", 1.5
        ds.Rows = ds.Columns = size
        ds.PixelSpacing = [0.5, 0.5]
        ds.ImageOrientationPatient = iop
        pos = [0.0, 0.0, 0.0]
        pos[axis] = 4.0 * k
        ds.ImagePositionPatient = pos
        ds.InstanceNumber = n_slices - k                      # deliberately not in spatial order
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.PixelData = rng.integers(0, 4000, (size, size), dtype=np.uint16).tobytes()
        try:
            ds.save_as(folder / f"{k:03d}.dcm", enforce_file_format=True)      # pydicom 3
        except TypeError:
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(folder / f"{k:03d}.dcm", write_like_original=False)


def make(root, n=5, seed=0):
    root, rng = Path(root), np.random.default_rng(seed)
    studies = [f"1.2.9.{100 + i}" for i in range(n)]
    rows = []
    for i, st in enumerate(studies):
        for j, plane in enumerate(PLANES):
            if plane == "Axial" and i == 1:
                continue                                       # a study without an axial series
            se = f"{st}.{j + 1}"
            write_series(root / "test_series" / st / se, plane, rng)
            rows.append({"StudyInstanceUID": st, "SeriesInstanceUID": se, "Anatomical_Plane": plane,
                         "Fluid_Sensitive": 1, "Fat_Suppression": 1})
    all_ids = studies + ["1.2.9.999"]                          # a study without any files
    pd.DataFrame(rows).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": all_ids}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": all_ids, **{l: 0.5 for l in LABELS}}).to_csv(root / "sample_submission.csv", index=False)
    tr = pd.DataFrame({"StudyInstanceUID": [f"1.2.8.{i}" for i in range(20)],
                       **{l: rng.integers(0, 2, 20).astype(float) for l in LABELS}})
    tr.to_csv(root / "train.csv", index=False)
    return root


if __name__ == "__main__":
    args = sys.argv[1:]
    print(make(args[0], *(int(a) for a in args[1:])))
