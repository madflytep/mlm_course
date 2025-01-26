"""
Functions to calculate scores for receipt requisites.

The top-level function is `derive_scores`.
"""

import datetime as dt
from decimal import Decimal
from itertools import product

import numpy as np
from rapidfuzz.fuzz import partial_ratio


def get_operation_datetime_variants(datetime: dt.datetime, country: str):
    if country == "ru":
        months = {
            "january": "января",
            "february": "февраля",
            "march": "марта",
            "april": "апреля",
            "may": "мая",
            "june": "июня",
            "july": "июля",
            "august": "августа",
            "september": "сентября",
            "october": "октября",
            "november": "ноября",
            "december": "декабря"
        }
        return [
            # Сбербанк "1 января 2021 года 00:00:00"
            datetime.strftime(f"%d {months[datetime.strftime('%B').lower()]} %Y года %H:%M:%S"),
            # Тинькофф, Альфа "01.01.2021 00:00:00"
            datetime.strftime(f"%d.%m.%Y %H:%M:%S"),
            # ВТБ "01.01.2021, 00:00"
            datetime.strftime(f"%d.%m.%Y, %H:%M"),
            # ОТП "01/01/2021 г."
            datetime.strftime(f"%d/%m/%Y г."),
            # Яндекс-банк "01.01.2021 в 00:00"
            datetime.strftime(f"%d.%m.%Y в %H:%M"),
        ]
    else:
        raise ValueError(f"Country '{country}' is not supported")


def get_operation_sum_variants(number: Decimal, country: str):

    def format_decimal(value, thousands_sep=" ", decimal_sep=",", ommit_fraction=False):
        int_part, frac_part = f"{value:.2f}".split(".")
        int_part = "{:,}".format(int(int_part)).replace(",", thousands_sep)
        if not ommit_fraction:
            return f"{int_part}{decimal_sep}{frac_part}"
        else:
            return f"{int_part}"
    
    def has_fraction(decimal_number):
        return decimal_number % 1 != 0
    
    if country == "ru":
        return [
            # Сбербанк, Яндекс-банк "15 110,00 ₽" or "15 110.00 ₽"
            format_decimal(number, thousands_sep=" ", decimal_sep=",") + " ₽",
            format_decimal(number, thousands_sep=" ", decimal_sep=".") + " ₽",
            # Тинькофф "110,42 ₽" or "110 ₽"
            format_decimal(number, thousands_sep=" ", decimal_sep=",",
                           ommit_fraction=not has_fraction(number)) + " ₽",
            # Альфа "1600 RUR" or "1 600.00 р."
            format_decimal(number, thousands_sep="", decimal_sep=".",
                           ommit_fraction=not has_fraction(number)) + " RUR",
            format_decimal(number, thousands_sep=" ", decimal_sep=".") + " р.",
            # Райффайзен "1600.00 ₽"
            format_decimal(number, thousands_sep="", decimal_sep=".") + " ₽",
            # ОТП "1600.00 RUB"
            format_decimal(number, thousands_sep="", decimal_sep=".") + " RUB",
            # Уральский банк "110.42 Руб" or "110 Руб"
            format_decimal(number, thousands_sep="", decimal_sep=".",
                           ommit_fraction=not has_fraction(number)) + " Руб",
        ]
    else:
        raise ValueError(f"Country '{country}' is not supported")
    
    
def get_recepient_account_variants(account: str, country: str):
    if country == "ru":
        return [account]
    else:
        raise ValueError(f"Country '{country}' is not supported")


def get_recepient_telnum_variants(telnum: str, country: str):
    if country == "ru":
        return [telnum]
    else:
        raise ValueError(f"Country '{country}' is not supported")


def match_field(variants, ocr_results, similarity_threshold=60):
    n_vars = len(variants)
    n_boxes = len(ocr_results)
    similarities = np.full((n_boxes, n_vars), np.nan)
    max_similarity = 0
    max_idx = None
    for i, j in product(range(n_boxes), range(n_vars)):
        similarity = partial_ratio(variants[j], ocr_results[i][1])
        similarities[i, j] = similarity
        if similarity > max_similarity:
            max_similarity = similarity
            max_idx = i
    if max_similarity >= similarity_threshold:
        return max_idx, max_similarity
    else:
        return None, max_similarity


def derive_scores(vlm_results, ocr_results, country):

    # TODO: Sometimes the operation datetime cosisted of two boxes
    # (date and time), need to join them
    
    def calculate_confidence(recognition_score, detection_score, distance_based_score):
        return (2 * recognition_score + 2 * detection_score + distance_based_score) / 5

    get_variants_functions = {
        "operation_datetime": get_operation_datetime_variants,
        "operation_sum": get_operation_sum_variants,
        "recepient_account": get_recepient_account_variants,
        "recepient_telnum": get_recepient_account_variants,
        "sender_bank": None,
    }

    result = {
        field: {
            "value": None,
            "origin_text": None,
            "origin_box": None,
            "scores": {
                "recognition": None,
                "detection": None,
                "distance_based": None,
                "confidence": None
            }
        }
        for field in vlm_results.keys()
    }
    
    for field in vlm_results.keys():
        if get_variants_functions[field] and vlm_results[field]:
            variants = get_variants_functions[field](vlm_results[field], country)
            matched_box_idx, similarity = match_field(variants, ocr_results)
            if matched_box_idx:
                result[field]["value"] = vlm_results[field]
                result[field]["origin_text"] = ocr_results[matched_box_idx][1]
                result[field]["origin_box"] = ocr_results[matched_box_idx][0]
                result[field]["scores"]["recognition"] = ocr_results[matched_box_idx][2]
                result[field]["scores"]["detection"] = ocr_results[matched_box_idx][3]
                result[field]["scores"]["distance_based"] = similarity / 100
                result[field]["scores"]["confidence"] = calculate_confidence(
                    ocr_results[matched_box_idx][2],
                    ocr_results[matched_box_idx][3],
                    similarity / 100
                )
            
    return result
