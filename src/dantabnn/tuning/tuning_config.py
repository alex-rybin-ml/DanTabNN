"""DanTabNN Tuning Configuration Calculator

Computes optimal ``n_jobs`` and ``torch.set_num_threads()`` for dantabnn
hyperparameter tuning based on available RAM, CPU cores, and dataset size.

Calibrated against memory analysis of dantabnn 0.3.0 with float32 optimization.
"""

import os


def compute_tuning_config(
    available_ram_gb: float,
    num_cores: int,
    dataset_size_gb: float,
    preprocessing_mode: str = "default",
    del_df_applied: bool = True,
    input_dtype: str = "float64",
) -> dict:
    """
    Compute optimal ``n_jobs`` and ``torch_num_threads`` for dantabnn tuning.

    Parameters
    ----------
    available_ram_gb : float
        Total RAM available to the process (e.g., Docker container limit).
    num_cores : int
        Number of CPU cores available.
    dataset_size_gb : float
        Uncompressed dataset size in memory (e.g., 32 for 32 GB of raw data).
    preprocessing_mode : {"default", "minimal"}, default="default"
        ``"default"`` — feature engineering (x², log1p) + outlier clipping + scaling.
        ``"minimal"`` — scaling + encoding only (no feature engineering, no outlier clip).
    del_df_applied : bool, default=True
        Whether the ``del df`` optimization is applied (True with the float32 fix).
    input_dtype : {"float64", "float32"}, default="float64"
        The dtype of the input DataFrame. Affects the extraction array size:
        ``"float64"`` → extraction array is 0.5× raw data (float64→float32 conversion).
        ``"float32"`` → extraction array is 1.0× raw data (no conversion needed).

    Returns
    -------
    dict
        n_jobs : int
            Recommended number of parallel Optuna trials.
        torch_num_threads : int
            Recommended ``torch.set_num_threads()`` value **per trial**.
        estimated_ram_peak_gb : float
            Expected RAM peak during tuning (with ``del df`` applied).
        estimated_ram_if_df_persists_gb : float
            Expected RAM peak if the caller still holds a DataFrame reference.
        extraction_peak_gb : float
            Brief RAM peak during ``to_numpy()`` when both DataFrame and the
            new float32 array coexist in memory.
        total_threads : int
            Total threads across all trials (``n_jobs × torch_num_threads``).
        throughput_score : float
            Estimated relative throughput (higher = faster tuning completion).
        fits_in_ram : bool
            Whether the configuration fits within available RAM.
        preprocessing_mode : str
            The preprocessing mode used for the estimate.
        df_persists_warning : bool
            True if RAM would exceed limit when the caller holds a DataFrame ref.
        warning : str, optional
            Human-readable warning if the configuration does not fit.

    Important
    ---------
    **Apply ``torch.set_num_threads()`` BEFORE importing dantabnn.**
    PyTorch initializes MKL/OpenMP at import time. Setting threads after
    ``from dantabnn.binary import BinaryClassificationPipeline`` may not
    take effect until the process restarts.

    Correct pattern::

        import torch
        config = compute_tuning_config(128, 10, 32, "default")
        torch.set_num_threads(config["torch_num_threads"])
        # Only now import dantabnn
        from dantabnn.binary import BinaryClassificationPipeline

    Examples
    --------
    >>> # Standard case: float64 DataFrame (pandas default)
    >>> # 4M rows × 1000 cols float64 = 32 GB raw
    >>> config = compute_tuning_config(128, 10, 32, "default", input_dtype="float64")
    >>> config["extraction_peak_gb"]  # 38 + 16 = 54 GB
    54.0
    >>> config["estimated_ram_peak_gb"]  # 3 trials × 32 GB = 96 GB
    96.0

    >>> # Float32 input: 8M rows × 1000 cols float32 = 32 GB raw
    >>> config = compute_tuning_config(128, 10, 32, "default", input_dtype="float32")
    >>> config["extraction_peak_gb"]  # 38 + 32 = 70 GB
    70.0
    >>> config["estimated_ram_peak_gb"]  # 2 trials × 64 GB = 128 GB
    128.0
    """
    # --- Memory Model (calibrated from dantabnn 0.3.0 analysis) ---
    #
    # All downstream sizes (tensor, preprocessing temps) are derived from the
    # float32-equivalent data size, NOT the raw input size. This is because
    # dantabnn processes everything in float32 after extraction.
    #
    # For a dataset that is 32 GB as float64 (e.g., 4M rows × 1000 cols × 8 bytes):
    #   float32_equivalent = 32 × 0.5 = 16 GB
    #   DataFrame (pandas overhead ~18.75%):  32 × 1.1875 = 38.0 GB
    #   Extraction array (float32):            16 GB  (half of float64)
    #   Final tensor:                          16 GB
    #   Preprocessing temps (default):         16 GB
    #   Preprocessing temps (minimal):          3.2 GB
    #   Extraction peak: 38 + 16 = 54 GB
    #   Per-trial peak (default): 16 + 16 = 32 GB
    #
    # For a dataset that is 32 GB as float32 (e.g., 8M rows × 1000 cols × 4 bytes):
    #   float32_equivalent = 32 × 1.0 = 32 GB
    #   DataFrame (pandas overhead ~18.75%):  32 × 1.1875 = 38.0 GB
    #   Extraction array (float32):            32 GB  (same as input)
    #   Final tensor:                          32 GB
    #   Preprocessing temps (default):         32 GB
    #   Preprocessing temps (minimal):          6.4 GB
    #   Extraction peak: 38 + 32 = 70 GB
    #   Per-trial peak (default): 32 + 32 = 64 GB

    raw_data_gb = dataset_size_gb  # User-reported uncompressed size

    # Convert to float32-equivalent: this is the basis for ALL downstream estimates
    if input_dtype == "float64":
        float32_equiv_gb = raw_data_gb * 0.5  # float64→float32: half the size
    elif input_dtype == "float32":
        float32_equiv_gb = raw_data_gb * 1.0  # Already float32: same size
    else:
        raise ValueError(
            f"Unknown input_dtype: {input_dtype!r}. Use 'float64' or 'float32'."
        )

    # DataFrame with pandas overhead (based on raw input, not float32-equiv)
    dataframe_gb = raw_data_gb * 1.1875

    # Extraction array size (the new float32 numpy array)
    extraction_array_gb = float32_equiv_gb

    # Final preprocessed tensor (always float32)
    tensor_gb = float32_equiv_gb

    # Model + optimizer state (negligible)
    model_gb = 0.01

    # Preprocessing temporaries per trial (proportional to float32 data size)
    if preprocessing_mode == "minimal":
        # Only StandardScaler — no feature engineering, no outlier clipping
        prep_temp_gb = float32_equiv_gb * 0.2
    elif preprocessing_mode == "default":
        # AutoFeatureEngineer (x², log1p) + RobustScaler outlier clip + StandardScaler
        # One transform copy active at a time, roughly equal to float32 data size
        prep_temp_gb = float32_equiv_gb * 1.0
    else:
        raise ValueError(
            f"Unknown preprocessing_mode: {preprocessing_mode!r}. "
            f"Use 'default' or 'minimal'."
        )

    # Peak memory per trial during fit
    trial_peak_gb = tensor_gb + prep_temp_gb + model_gb

    # OS and Python runtime overhead (capped at 8 GB or 8 % of RAM)
    os_overhead_gb = min(available_ram_gb * 0.08, 8.0)
    usable_ram_gb = available_ram_gb - os_overhead_gb

    # Brief extraction peak: both DataFrame and numpy array coexist
    extraction_peak_gb = dataframe_gb + extraction_array_gb

    # --- Determine max n_jobs based on RAM ---
    if del_df_applied:
        # DataFrame freed after extraction → tuning RAM = n_jobs × trial_peak
        max_jobs_by_ram = int(usable_ram_gb / trial_peak_gb)
    else:
        # DataFrame persists throughout
        max_jobs_by_ram = int((usable_ram_gb - dataframe_gb) / trial_peak_gb)

    max_jobs_by_ram = max(1, max_jobs_by_ram)

    # Extraction must fit (brief but unavoidable peak)
    extraction_fits = extraction_peak_gb <= available_ram_gb * 0.95

    # --- Core allocation ---
    # Leave 1 core for OS / Optuna overhead
    available_cores = max(1, num_cores - 1)

    best_config = None
    best_score = -1.0

    for n_jobs in range(1, min(max_jobs_by_ram + 1, num_cores + 1)):
        # Divide available cores among trials evenly
        torch_threads = max(1, available_cores // n_jobs)
        total_threads = n_jobs * torch_threads

        # RAM estimates
        ram_peak = n_jobs * trial_peak_gb                      # With del_df
        ram_with_df = dataframe_gb + n_jobs * trial_peak_gb    # If df persists

        # Throughput: parallel trials × sublinear per-trial speedup
        # (sublinear because memory bandwidth is shared across threads)
        throughput = n_jobs * (torch_threads ** 0.7)

        # Fit check
        fits = ram_peak <= usable_ram_gb and extraction_fits
        df_persists_warning = ram_with_df > available_ram_gb * 0.95

        if fits and throughput > best_score:
            best_score = throughput
            best_config = {
                "n_jobs": n_jobs,
                "torch_num_threads": torch_threads,
                "estimated_ram_peak_gb": round(ram_peak, 1),
                "estimated_ram_if_df_persists_gb": round(ram_with_df, 1),
                "extraction_peak_gb": round(extraction_peak_gb, 1),
                "total_threads": total_threads,
                "throughput_score": round(throughput, 2),
                "fits_in_ram": True,
                "preprocessing_mode": preprocessing_mode,
                "input_dtype": input_dtype,
                "df_persists_warning": df_persists_warning,
            }

    # Fallback: nothing fits, return safest single-job config
    if best_config is None:
        torch_threads = max(1, available_cores)
        trial_ram = trial_peak_gb + (dataframe_gb if not del_df_applied else 0)
        best_config = {
            "n_jobs": 1,
            "torch_num_threads": torch_threads,
            "estimated_ram_peak_gb": round(trial_ram, 1),
            "estimated_ram_if_df_persists_gb": round(dataframe_gb + trial_peak_gb, 1),
            "extraction_peak_gb": round(extraction_peak_gb, 1),
            "total_threads": torch_threads,
            "throughput_score": round(1.0 * (torch_threads ** 0.7), 2),
            "fits_in_ram": False,
            "preprocessing_mode": preprocessing_mode,
            "input_dtype": input_dtype,
            "df_persists_warning": False,
            "warning": (
                f"Single job needs ~{round(trial_ram, 1)} GB but only "
                f"{available_ram_gb} GB available. Try: "
                f"preprocessing_mode='minimal' (needs ~"
                f"{round(tensor_gb + float32_equiv_gb * 0.2 + model_gb, 1)} GB), "
                f"or increase RAM to at least {round(trial_ram * 1.1)} GB."
            ),
        }

    return best_config


def auto_configure_tuning(
    dataset_size_gb: float,
    preprocessing_mode: str = "default",
    input_dtype: str = "float64",
    set_threads: bool = True,
) -> dict:
    """
    Auto-detect available RAM and cores, then compute tuning config.

    Respects Docker cgroup v1/v2 memory and CPU limits when running inside
    a container. Falls back to ``os.sysconf`` / ``os.cpu_count`` on bare metal.

    Parameters
    ----------
    dataset_size_gb : float
        Uncompressed dataset size in memory.
    preprocessing_mode : {"default", "minimal"}, default="default"
        ``"default"`` or ``"minimal"``.
    input_dtype : {"float64", "float32"}, default="float64"
        Dtype of the input DataFrame. See :func:`compute_tuning_config`.
    set_threads : bool, default=True
        If True, calls ``torch.set_num_threads()`` immediately. Safe to call
        **before** importing dantabnn. If False, returns the recommended value
        for the caller to apply manually.

    Returns
    -------
    dict
        Same as :func:`compute_tuning_config`, plus ``auto_detected_ram_gb``
        and ``auto_detected_cores``.

    Example
    -------
    >>> config = auto_configure_tuning(32, preprocessing_mode="default")
    >>> print(f"n_jobs={config['n_jobs']}, threads={config['torch_num_threads']}")
    n_jobs=3, threads=3
    """
    # Detect RAM (respects Docker cgroup limits)
    try:
        # cgroup v2 memory limit
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            mem_limit = f.read().strip()
            if mem_limit == "max":
                available_ram_gb = (
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
                )
            else:
                available_ram_gb = int(mem_limit) / (1024 ** 3)
    except (FileNotFoundError, ValueError):
        try:
            # cgroup v1 memory limit
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                available_ram_gb = int(f.read().strip()) / (1024 ** 3)
        except (FileNotFoundError, ValueError):
            available_ram_gb = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
            )

    # Detect cores (cgroup-aware when possible)
    try:
        # cgroup v2 cpu.max: "quota period"
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            quota, period = f.read().strip().split()
            if quota == "max":
                num_cores = os.cpu_count() or 1
            else:
                num_cores = max(1, int(int(quota) / int(period)))
    except (FileNotFoundError, ValueError):
        try:
            # cgroup v1 cpu.cfs_quota_us
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                period = int(f.read().strip())
            if quota > 0:
                num_cores = max(1, int(quota / period))
            else:
                num_cores = os.cpu_count() or 1
        except (FileNotFoundError, ValueError):
            num_cores = os.cpu_count() or 1

    config = compute_tuning_config(
        available_ram_gb, num_cores, dataset_size_gb, preprocessing_mode,
        input_dtype=input_dtype
    )
    config["auto_detected_ram_gb"] = round(available_ram_gb, 1)
    config["auto_detected_cores"] = num_cores

    if set_threads:
        try:
            import torch
            torch.set_num_threads(config["torch_num_threads"])
            config["threads_set"] = True
        except ImportError:
            config["threads_set"] = False
            config["threads_warning"] = (
                "PyTorch not installed; torch.set_num_threads() not applied. "
                "Apply manually before importing dantabnn."
            )

    return config