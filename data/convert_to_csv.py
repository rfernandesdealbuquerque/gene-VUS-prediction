import sys

import config
import pandas as pd


def convert_variants_to_csv(gene_name): 

    for variant_type in ['Benign_Likely Benign', 'Pathogenic_Likely Pathogenic', 'Uncertain Significance']:
        # Read the data from the specified sheet
        df = pd.read_excel(config.data + f'/{gene_name}/1. Raw/{gene_name}_concise_UpdatedGnomAD.xlsx', sheet_name=variant_type)

        # Process the data and save it as a CSV file (similar to the previous example)
        original = []
        position = []
        change = []

        for index, row in df.iterrows():
            protein_change = row['Protein change']
            parts = protein_change.split(',')
            # print(parts)
            for part in parts:
                part = part.strip()
                original.append(part[0])
                position.append(part[1:-1])
                change.append(part[-1])

        # Create a new DataFrame with the processed data
        processed_df = pd.DataFrame({'Original': original, 'AA position': position, 'Change': change})

        # Save the DataFrame as a CSV file
        if variant_type == 'Uncertain Significance':
            processed_df.to_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Uncertain_Variants.csv', index=False)
        else:
            processed_df.to_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_L{variant_type[0]}_{variant_type[0]}_Variants.csv', index=False)

    return

def convert_signal_to_noise_to_csv(gene_name):
    # Read the data from the specified sheet
    df = pd.read_excel(config.data + f'/{gene_name}/1. Raw/{gene_name}_concise_UpdatedGnomAD.xlsx', sheet_name='Signal to Noise @ AA position')
    df = df.iloc[1:, :]
    df.to_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Signal_to_Noise.csv', index=False)

    return


def convert_evolutionary_score_to_csv(gene_name):
    df = pd.read_excel(config.data + f'/{gene_name}/1. Raw/{gene_name}_concise_UpdatedGnomAD.xlsx', sheet_name='Evol. Conservation ID Score')
    df.to_csv(config.data + f'/{gene_name}/1. Raw/{gene_name}_Conservation_Score.csv', index=False)


gene_list = {'MYH7', 'RYR2'}

for gene in gene_list:
    convert_variants_to_csv(gene)
    convert_signal_to_noise_to_csv(gene)
    convert_evolutionary_score_to_csv(gene)