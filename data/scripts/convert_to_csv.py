import sys

import pandas as pd

sys.path.append('../../')
import config


def convert_variants_to_csv(): 

    # Read the data from the specified sheet
    df = pd.read_excel(config.data + '/KCNQ1/1. Raw/KCNQ1_concise_Updated GnomAD.xlsx', sheet_name='Benign_Likely Benign')

    # Process the data and save it as a CSV file (similar to the previous example)
    original = []
    position = []
    change = []

    for index, row in df.iterrows():
        protein_change = row['Protein change']
        parts = protein_change.split(',')
        print(parts)
        for part in parts:
            part = part.strip()
            original.append(part[0])
            position.append(part[1:-1])
            change.append(part[-1])

    # Create a new DataFrame with the processed data
    processed_df = pd.DataFrame({'Original': original, 'AA position': position, 'Change': change})

    # Save the DataFrame as a CSV file
    processed_df.to_csv(config.data + '/KCNQ1/1. Raw/KCNQ1_LB_B_Variants.csv', index=False)

    return

def convert_signal_to_noise_to_csv():
    # Read the data from the specified sheet
    df = pd.read_excel(config.data + '/KCNQ1/1. Raw/KCNQ1_concise_Updated GnomAD.xlsx', sheet_name='Signal to Noise @ AA position')
    df.to_csv(config.data + '/KCNQ1/1. Raw/KCNQ1_Signal_to_Noise.csv', index=False)

    return


def convert_evolutionary_score_to_csv():
    df = pd.read_excel(config.data + '/KCNQ1/1. Raw/KCNQ1_concise_Updated GnomAD.xlsx', sheet_name='Evol. Conservation ID Score')
    df.to_csv(config.data + '/KCNQ1/1. Raw/KCNQ1_Conservation_Score.csv', index=False)

convert_variants_to_csv()
convert_signal_to_noise_to_csv()
convert_evolutionary_score_to_csv()