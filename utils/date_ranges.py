"""Date range utilities for TRAFFIX with POC constraints."""

from datetime import datetime, timedelta
from typing import Dict, Tuple
from dateutil.relativedelta import relativedelta

from utils.config_loader import get_config


def get_poc_constraints() -> Tuple[datetime, datetime]:
    """
    Get POC date constraints from config.
    
    Returns:
        Tuple of (start_date, end_date) for POC period
    """
    config = get_config()
    poc_config = config.get("poc_date_range", {})
    
    start_str = poc_config.get("start", "2025-10-19")
    end_str = poc_config.get("end", "2025-11-02")
    
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    
    return start_date, end_date


def constrain_to_poc(start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
    """
    Constrain date range to POC period.
    
    Args:
        start_date: Desired start date
        end_date: Desired end date
        
    Returns:
        Tuple of (constrained_start, constrained_end)
    """
    poc_start, poc_end = get_poc_constraints()
    
    # Clamp to POC range
    constrained_start = max(start_date, poc_start)
    constrained_end = min(end_date, poc_end)
    
    return constrained_start, constrained_end


def get_mtd_range(reference_date: datetime = None) -> Dict[str, datetime]:
    """
    Get Month-to-Date range.
    
    Args:
        reference_date: Reference date (defaults to POC end date)
        
    Returns:
        Dictionary with 'start' and 'end' datetime objects
    """
    if reference_date is None:
        _, poc_end = get_poc_constraints()
        reference_date = poc_end
        
    # Start of current month
    start_date = reference_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = reference_date
    
    # Constrain to POC
    start_date, end_date = constrain_to_poc(start_date, end_date)
    
    return {
        "start": start_date,
        "end": end_date,
        "label": "MTD",
        "description": f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
    }


def get_qtd_range(reference_date: datetime = None) -> Dict[str, datetime]:
    """
    Get Quarter-to-Date range.
    
    Args:
        reference_date: Reference date (defaults to POC end date)
        
    Returns:
        Dictionary with 'start' and 'end' datetime objects
    """
    if reference_date is None:
        _, poc_end = get_poc_constraints()
        reference_date = poc_end
        
    # Determine quarter start month
    quarter_start_month = ((reference_date.month - 1) // 3) * 3 + 1
    start_date = reference_date.replace(
        month=quarter_start_month,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )
    end_date = reference_date
    
    # Constrain to POC
    start_date, end_date = constrain_to_poc(start_date, end_date)
    
    return {
        "start": start_date,
        "end": end_date,
        "label": "QTD",
        "description": f"Q{(reference_date.month - 1) // 3 + 1} {reference_date.year}"
    }


def get_ytd_range(reference_date: datetime = None) -> Dict[str, datetime]:
    """
    Get Year-to-Date range.
    
    Args:
        reference_date: Reference date (defaults to POC end date)
        
    Returns:
        Dictionary with 'start' and 'end' datetime objects
    """
    if reference_date is None:
        _, poc_end = get_poc_constraints()
        reference_date = poc_end
        
    # Start of current year
    start_date = reference_date.replace(
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )
    end_date = reference_date
    
    # Constrain to POC
    start_date, end_date = constrain_to_poc(start_date, end_date)
    
    return {
        "start": start_date,
        "end": end_date,
        "label": "YTD",
        "description": f"YTD {reference_date.year}"
    }


def get_last_n_days(n_days: int, reference_date: datetime = None) -> Dict[str, datetime]:
    """
    Get last N days range.
    
    Args:
        n_days: Number of days to go back
        reference_date: Reference date (defaults to POC end date)
        
    Returns:
        Dictionary with 'start' and 'end' datetime objects
    """
    if reference_date is None:
        _, poc_end = get_poc_constraints()
        reference_date = poc_end
        
    end_date = reference_date
    start_date = end_date - timedelta(days=n_days)
    
    # Constrain to POC
    start_date, end_date = constrain_to_poc(start_date, end_date)
    
    return {
        "start": start_date,
        "end": end_date,
        "label": f"Last {n_days} Days",
        "description": f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
    }


def get_custom_range(start_date: datetime, end_date: datetime) -> Dict[str, datetime]:
    """
    Get custom date range with POC constraints.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Dictionary with 'start' and 'end' datetime objects
    """
    # Constrain to POC
    start_date, end_date = constrain_to_poc(start_date, end_date)
    
    return {
        "start": start_date,
        "end": end_date,
        "label": "Custom",
        "description": f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
    }


def get_date_range_by_type(
    range_type: str,
    custom_start: datetime = None,
    custom_end: datetime = None,
    reference_date: datetime = None
) -> Dict[str, datetime]:
    """
    Get date range by type.
    
    Args:
        range_type: One of 'MTD', 'QTD', 'YTD', 'Last7', 'Last30', 'Custom'
        custom_start: For custom range
        custom_end: For custom range
        reference_date: Reference date for calculations
        
    Returns:
        Dictionary with 'start' and 'end' datetime objects
    """
    range_type = range_type.upper()
    
    if range_type == "MTD":
        return get_mtd_range(reference_date)
    elif range_type == "QTD":
        return get_qtd_range(reference_date)
    elif range_type == "YTD":
        return get_ytd_range(reference_date)
    elif range_type == "LAST7":
        return get_last_n_days(7, reference_date)
    elif range_type == "LAST30":
        return get_last_n_days(30, reference_date)
    elif range_type == "CUSTOM":
        if custom_start and custom_end:
            return get_custom_range(custom_start, custom_end)
        else:
            # Default to last 7 days
            return get_last_n_days(7, reference_date)
    else:
        # Default to last 7 days
        return get_last_n_days(7, reference_date)


# Predefined range options for UI
DATE_RANGE_OPTIONS = [
    {"value": "LAST7", "label": "Last 7 Days"},
    {"value": "LAST30", "label": "Last 30 Days"},
    {"value": "MTD", "label": "Month to Date"},
    {"value": "QTD", "label": "Quarter to Date"},
    {"value": "YTD", "label": "Year to Date"},
    {"value": "CUSTOM", "label": "Custom Range"},
]

