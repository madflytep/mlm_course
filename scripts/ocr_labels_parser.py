import json
from typing import Any, Dict

import click
import pandas as pd


def parse_unicode(s: str) -> str:
    """Convert Unicode escape sequences to UTF-8."""
    return str(s).encode("latin1").decode("unicode_escape")


def safe_json_loads(json_string: str) -> Dict[str, Any]:
    """Attempt to parse JSON and return the result or log an error if parsing fails."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        print(
            f"Error parsing JSON: {e} - Input: {json_string} -> error strings will be returned"
        )
        return {"ERROR": "ERROR"}


def parse_json_column(df: pd.DataFrame, json_column: str, suffix="") -> pd.DataFrame:
    """Parse the JSON-like column and return a new DataFrame with the parsed data."""
    # Convert the column to UTF-8 and parse JSON
    df[json_column] = df[json_column].apply(parse_unicode)
    # parsed_data = df[source_column].apply(safe_json_loads)  #  Instead of json.loads
    parsed_data = df[json_column].apply(safe_json_loads).tolist()

    # Ensure all items are dictionaries
    parsed_data = [
        item if isinstance(item, dict) else {"ERROR": "ERROR"} for item in parsed_data
    ]

    # Add a suffix to the field names if specified
    if suffix:
        parsed_data = [{k + "_" + suffix: v for k, v in row.items()}
                       for row in parsed_data]

    # Normalize JSON data into a DataFrame
    parsed_df = pd.json_normalize(parsed_data)

    # Combine with original DataFrame if needed
    result_df = df.join(parsed_df)

    return result_df


@click.command()
@click.argument("input_excel_filename", type=click.Path(exists=True))
@click.argument("output_excel_filename", type=click.Path())
@click.argument("ocr_column", type=str)
@click.argument("match_column", type=str)
def main(
    input_excel_filename: str,
    output_excel_filename: str,
    ocr_column: str,
    match_column: str
):
    """Parse JSON-like column in Excel file and output to a new Excel file."""
    # Read the Excel file
    df = pd.read_excel(input_excel_filename, skiprows=1)

    # Parse the column and get the output DataFrame
    result_df = parse_json_column(df, ocr_column)
    result_df = parse_json_column(result_df, match_column, suffix="deal")

    # Save the result to a new Excel file
    result_df.to_excel(output_excel_filename, index=False)
    click.echo(f"Parsed data saved to {output_excel_filename}")


if __name__ == "__main__":
    main()
