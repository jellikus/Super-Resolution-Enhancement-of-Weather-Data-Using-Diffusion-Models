import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, ConcatDataset
from configs.config import DataConfig

config = DataConfig()
DATETIME_FORMAT = config.datetime_format


def check_valid_format(date: str) -> bool:
    """Checks if date is in a valid format.

    Args:
        date: Date in string format.

    Returns:
        True if date is in a valid format, False otherwise.
    """
    try:
        datetime.strptime(date, DATETIME_FORMAT)
        return True
    except ValueError:
        return False


def get_month_idx(date: str) -> int:
    """Returns month int index from a given date.

    Args:
        date: Date in datetime format.

    Returns:
        Month index.
    """
    assert check_valid_format(date), f"Date {date} is not in a valid format"
    date = str_to_date(date)
    return date.month


def date_to_str(datetime_object: datetime) -> str:
    """Converts datetime object into string.

    Args:
        datetime_object: Time representation.

    Returns:
        Returns string date.
    """
    return datetime.strftime(datetime_object, DATETIME_FORMAT)


def str_to_date(date: str) -> datetime:
    """Converts string date into datetime object.

    Args:
        date: Date in string format.

    Returns:
        Returns datetime object.
    """
    return datetime.strptime(date, DATETIME_FORMAT)


def get_month_datetime():
    """Returns a datetime object representing a month.

    Returns :
        A datetime object.
    """
    return relativedelta(months=1)


def find_group_idx(month: int, groups: list) -> int or None:
    """Finds group index for a given month.

    Args:
        month: Month index.
        groups: A list of groups of months.

    Returns:
        Group index.
    """
    for idx, group in enumerate(groups):
        if month in group:
            return idx + 1

    return None


def is_full_year(months_subset: list or None):
    """Checks if months_subset is a full year.

    Args:
        months_subset: A list of months.

    Returns:
        True if months_subset is a full year, False otherwise.
    """
    if months_subset is None:
        return True
    months_subset = set(months_subset)
    year = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    return months_subset == year


def is_group_full_year(groups):
    """Checks if groups is a full year.

    Args:
        groups: A list of groups of months.

    Returns:
        True if groups is a full year, False otherwise.
    """
    if groups is None:
        return False
    if len(groups) == 1:
        return set(groups[0]) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    return False


def validate_group_months_subset(months_subset, groups):
    """
    Validate the relationship between a specified months_subset and groups of months.

    Args:
        months_subset (list or None): List of months to validate against (1-12).
            If None, expects groups to cover all months from 1 to 12.
        groups (list of lists): Nested list of months grouped together.

    Raises:
        AssertionError: If validation checks fail.
    """

    assert months_subset is not None or groups is not None, "months_subset and groups cannot be both None"
    year = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

    groups = [item for sublist in groups for item in sublist]
    assert len(groups) <= 12, f"groups {groups} has more than 12 months"

    if months_subset is None:
        assert year == set(
            groups), f"groups missing some months {groups} from 1 to 12"
    else:
        assert set(months_subset).issubset(year), f"months_subset {months_subset} does not contain valid months"
    assert len(months_subset) == len(groups), f"months_subset {months_subset} has different len than {groups}"
    assert set(months_subset) == set(
        groups), f"months_subset {months_subset} does not contain same numbers as groups {groups}"


def validate_month_subset(months_subset):
    """
    Validate a given months_subset against the standard set of months (1-12).

    Args:
        months_subset (list or None): List of months to validate.

    Returns:
        bool: True if months_subset is None or contains valid month numbers (1-12).

    Raises:
        AssertionError: If months_subset contains invalid month numbers.
    """
    year = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    if months_subset is None:
        return True
    assert set(months_subset).issubset(year), f"months_subset {months_subset} does not contain valid months"


def unpack_datasets(datasets) -> list:
    """Unpacks a concatenated datasets and creates alist of those datasets.

    Args:
        datasets: Either ConcatDataset object or TimeVariateData.

    Returns:
        A list of TimeVariateData datasets. If datasets is TimeVariateData,
        object, returns that object NOT in a list.
    """
    return [unpack_datasets(dataset) for dataset in datasets.datasets] \
        if isinstance(datasets, ConcatDataset) else datasets


def flatten(list_of_lists):
    """Flattens a nested-list structure.

    Args:
        list_of_lists: A list of nested lists.

    Returns:
        Flattened 1-dimensional list.
    """
    if not isinstance(list_of_lists, list):
        return [list_of_lists]
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def save_object(obj, path: str, filename: str) -> None:
    """Saves python object with pickle.

    Args:
        obj: Object to save.
        path: A directory where to save.
        filename: The name of a file in which to write.
    """
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"

    with open(os.path.join(path, filename), "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def log_dataset_info(dataset: Dataset, dataset_name: str, logger) -> None:
    """Logs dataset information.

    Args:
        dataset: A pytorch dataset.
        dataset_name: The name of dataset.
        logger: Logging object.
    """
    logger.info(f"Dataset [{dataset.__class__.__name__} - {dataset_name}] is created.")
    logger.info(f"""Created {dataset.__class__.__name__} dataset of length {len(dataset)}, containing data 
    from {dataset.min_date} until {dataset.max_date}""")
    logger.info(f"Group structure: {dataset.get_data_names()}")
    logger.info(f"Channel count: {dataset.get_channel_count()}")
    logger.info(f"Dataset size: {len(dataset)}\n")
