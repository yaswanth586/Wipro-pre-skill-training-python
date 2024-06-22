# Load Datasets from Various File Formats
# Load a CSV file named data.csv into a Pandas DataFrame.
# Load a JSON file named data.json into a Pandas DataFrame.
# Load an Excel file named data.xlsx into a Pandas DataFrame.
# Ask how many rows to display from the DataFrame and display the results


import pandas as pd


def write_data():
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [28, 34, 22, 35, 29],
        'score': [85, 90, 95, 88, 92]
    })

    data.to_csv('data.csv', index=False)
    data.to_excel('data.xlsx', index=False)
    data.to_json('data.json', orient='records')


def load_data():
    csv_df = pd.read_csv('data.csv')
    json_df = pd.read_json('data.json')
    excel_df = pd.read_excel('data.xlsx')

    return csv_df, json_df, excel_df


if __name__ == "__main__":
    num_rows = int(input("How many rows would you like to display? "))

    write_data()

    csv_df, json_df, excel_df = load_data()

    print("CSV DataFrame:\n", csv_df.head(num_rows))
    print("JSON DataFrame:\n", json_df.head(num_rows))
    print("Excel DataFrame:\n", excel_df.head(num_rows))
