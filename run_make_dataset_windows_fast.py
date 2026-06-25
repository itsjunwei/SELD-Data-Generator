# run_make_dataset_windows_fast.py

import os
import sys
import multiprocessing as mp

# Must be set before NumPy/SciPy are imported by the generator.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from make_dataset import main


if __name__ == "__main__":
    mp.freeze_support()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_make_dataset_windows_fast.py <task_id>")

    main(["make_dataset.py", sys.argv[1]])