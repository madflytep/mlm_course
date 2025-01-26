import math
import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import dateparser
from loguru import logger


def parse_datetime(text: str|None) -> dt.datetime|None:
    if text is None:
        return None
    if isinstance(text, float) and math.isnan(text):
        return None
    try:
        datetime = dt.datetime.strptime(text, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        datetime = None
    if datetime is None:
        try:
            datetime = dt.datetime.strptime(text, "%d.%m.%Y %H:%M")
        except ValueError:
            datetime = None
    if datetime is None:
        datetime = dateparser.parse(text)
    if datetime is None:
        logger.debug(f"Failed to parse datetime: {text}")
    return datetime


def parse_sum(text: str|None) -> Decimal|None:
    if text is None:
        return None
    cleaned_text = re.sub(r"[^0-9.]", "", text)
    try:
        parsed_sum = Decimal(cleaned_text)
    except InvalidOperation as e:
        logger.debug(f"Failed to parse sum:\n"
                     f"text={text} cleaned_text={cleaned_text}")
        parsed_sum = None
    return parsed_sum


def parse_account(text: str|None) -> str|None:
    if text is None:
        return None
    
    # Spectial case for "ПриватБанк"
    # For accounts like "535528********23" remain onle last 2 digits.
    last_four_digits = text[-4:]
    if (
        len(last_four_digits) == 4 
        and last_four_digits.startswith("**")
        and last_four_digits[-2:].isdigit()
    ):
        return last_four_digits[-2:]
    
    cleaned_account = re.sub(r"[^0-9]", "", text)
    
    return cleaned_account


def parse_telnum(text: str|None) -> str|None:
    if text is None:
        return None
    
    # There are special cases for "ОТП Банк" and "Альфа-Банк,"
    # which use masked telephone numbers like "+7982*****56" and "996***1910,"
    # respectively. To support them, '*' should not be removed.
    
    cleaned_account = re.sub(r"[^0-9+*]", "", text)
    
    return cleaned_account


def extract_field_value(line: str, field_name: str) -> str|None:
    field_name = re.escape(field_name)
    pattern = rf"{field_name}:\s*(.+)"
    match = re.search(pattern, line)
    if match:
        return match.group(1).strip()
    else:
        return None


def check_telnum(telnum: str|None) -> str|None:
    if telnum is None:
        return None
    if telnum.startswith("+"):
        return telnum if len(telnum) == 12 else None
    else:
        return telnum if len(telnum) == 11 else None


def parse_response_named_fields(response: str) -> dict[str, Any]:
    
    lines = response.split("\n")
    if not (4 <= len(lines) <= 5):
        logger.debug(f"Invalid number of lines: {len(lines)}\n"
                     f"LM response: {response}")

    parsed = {
        "operation_datetime": None,
        "operation_sum": None,
        "recepient_account": None,
        "recepient_telnum": None,
        "sender_bank": None
    }

    for line in lines:
        for field_name in parsed.keys():
            value = extract_field_value(line, field_name)
            if value is not None:
                if parsed[field_name] is not None:
                    logger.debug(f"Duplicate field in LM response: {field_name}\n"
                                 f"Old value: {parsed[field_name]}\n"
                                 f"New value will be assigned: {value}\n"
                                 f"LM response: {response}")
                parsed[field_name] = value
    
    parsed["operation_datetime"] = parse_datetime(parsed["operation_datetime"])
    parsed["operation_sum"] = parse_sum(parsed["operation_sum"])
    parsed["recepient_account"] = parse_account(parsed["recepient_account"])
    parsed["recepient_telnum"] = parse_telnum(parsed["recepient_telnum"])
    
    return parsed


def check_named_fields(
    parsed,
    sender_bank,
    operation_datetime,
    operation_sum,
    recepient_account,
    recepient_telnum
):
    
    matches = {
        "sender_bank": False,
        "operation_datetime": False,
        "operation_sum": False,
        "recepient_account": False,
        "recepient_telnum": False
    }
    
    # #
    # Try to match the sender bank
    
    matches["sender_bank"] = sender_bank == parsed["sender_bank"]
    
    if not matches["sender_bank"]:
        logger.debug(f"Failed to match sender_bank:\n"
                     f"parsed={parsed['sender_bank']}\n"
                     f"ground_truth={sender_bank}")
    
    # #
    # Try to match the date
    
    # Discard seconds from ground truth datetime
    operation_datetime = parse_datetime(operation_datetime)
    if operation_datetime is not None:
        operation_datetime = operation_datetime.strftime("%d.%m.%Y %H:%M")
    else:
        if parsed["operation_datetime"] is None:
            matches["operation_datetime"] = True
    
    # Discard seconds from parsed datetime
    if not matches["operation_datetime"] and parsed["operation_datetime"]:
        if parsed["operation_datetime"]:
            parsed["operation_datetime"] = parsed["operation_datetime"].strftime("%d.%m.%Y %H:%M")
            matches["operation_datetime"] = operation_datetime == parsed["operation_datetime"]
    
    if not matches["operation_datetime"]:
        logger.debug(f"Failed to match operation_datetime:\n"
                     f"parsed={parsed['operation_datetime']}\n"
                     f"ground_truth={operation_datetime}")
    
    # #
    # Try to match the recepient account

    if parsed["recepient_account"] is not None:
        matches["recepient_account"] = recepient_account == parsed["recepient_account"]
        if len(parsed["recepient_account"]) == 2:  # Special case for "ПриватБанк"
            if not (isinstance(recepient_account, float) and math.isnan(recepient_account)):
                matches["recepient_account"] = recepient_account.endswith(parsed["recepient_account"])
            else:  # The GT is float('nan') actually
                matches["recepient_account"] = False
    else:
        if isinstance(recepient_account, float) and math.isnan(recepient_account):  # The GT is also NaN
            matches["recepient_account"] = True
        

    if not matches["recepient_account"]:
        logger.debug(f"Failed to match recepient_account:\n"
                     f"parsed={parsed['recepient_account']}\n"
                     f"ground_truth={recepient_account}")
    
    # #
    # Try to match the recepient telnum
    
    if parsed["recepient_telnum"] is not None:
        matches["recepient_telnum"] = recepient_telnum == parsed["recepient_telnum"]
    else:
        if isinstance(recepient_telnum, float) and math.isnan(recepient_telnum):
            matches["recepient_telnum"] = True
    
    if not matches["recepient_telnum"]:
        logger.debug(f"Failed to match recepient_telnum:\n"
                     f"parsed={parsed['recepient_telnum']}\n"
                     f"ground_truth={recepient_telnum}")
    
    # #
    # Try to match the sum
    
    if parsed["operation_sum"] is not None:
        matches["operation_sum"] = Decimal(operation_sum) == parsed["operation_sum"]
    else:  # check if both are NaN
        # For `float`:
        if isinstance(operation_sum, float) and math.isnan(operation_sum):
            matches["operation_sum"] = True
        # For `Decimal`:
        if isinstance(operation_sum, Decimal) and operation_sum.is_nan():
            matches["operation_sum"] = True

    if not matches["operation_sum"]:
        logger.debug(f"Failed to match operation_sum:\n"
                     f"parsed={parsed['operation_sum']}\n"
                     f"ground_truth={operation_sum}")

    return matches
