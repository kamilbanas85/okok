#-----------------------------------------------------------------------------
# function to return dictionary: train_start, test_start, train_period
#-----------------------------------------------------------------------------

def build_train_test_windows(
    index: pd.DatetimeIndex,
    test_days_nr: int,
    train_period_list: list
) -> list[dict]:
    """
    Build a list of train/test window specs from a mixed list of
    training periods, with test_start derived from the data's
    DatetimeIndex and the test window length.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The DatetimeIndex of the dataset (e.g. X.index).
    test_days_nr : int
        Number of days held out for testing.
    train_period_list : list
        Each element is either:
            - int  : number of months to look back from `test_start`
                     to compute `train_start` (rolling window).
            - str  : an explicit train_start date (e.g. '2020-04-09'),
                     used as-is for both `train_start` and `train_period`
                     (expanding window).
    date_format : str
        Output string format for dates. Default '%Y-%m-%d'.

    Returns
    -------
    list[dict]
        [
            {'train_start': ..., 'test_start': ..., 'train_period': ...},
            ...
        ]
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"index must be a pandas DatetimeIndex, got {type(index)}"
        )

    if index.empty:
        raise ValueError("index is empty; cannot determine last available date.")

    last_data_ts = index.max()

    # Derive test_start from the data + test window length
    test_start_ts = last_data_ts.normalize() - pd.Timedelta(days=test_days_nr)
    test_start_str = test_start_ts.strftime("%Y-%m-%d")

    windows = []

    for train_period in train_period_list:

        if isinstance(train_period, int):
            # Rolling window: subtract N months from test_start
            train_start_ts = test_start_ts - pd.DateOffset(months=train_period)
            train_start_str = train_start_ts.strftime("%Y-%m-%d")

        elif isinstance(train_period, str):
            # Expanding window: explicit fixed start date
            train_start_ts = pd.Timestamp(train_period)  # validates format
            train_start_str = train_start_ts.strftime("%Y-%m-%d")

        else:
            raise TypeError(
                f"train_period entries must be int (months) or str (date), "
                f"got {type(train_period)}: {train_period!r}"
            )

        windows.append({
            "train_start": train_start_str,
            "test_start": test_start_str,
            "train_period": train_period,
        })

    return windows
