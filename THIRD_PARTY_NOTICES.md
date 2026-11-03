# Third-party software and data

The frozen input release includes the exact local V2SIM source used for the experiments, its original BSD-3-Clause LICENSE, case files and project metadata. Upstream: https://github.com/hesl-seu/v2sim . Its embedded simulation components retain their upstream notices.

NumPy, SciPy/HiGHS, PyTorch, pandapower, fpowerkit, feasytools and other dependencies retain their respective licenses. They are installed from their package sources; Python environments and credentials are not included.

Wind, solar and load traces are the existing calibrated research inputs. Their local provenance and normalization metadata are retained in `artifacts/data/reference.json` and the frozen manifest. They should be described as calibrated simulation inputs rather than city-level measured operational data.

No new license is assigned to the original TARP research code by this upload.
