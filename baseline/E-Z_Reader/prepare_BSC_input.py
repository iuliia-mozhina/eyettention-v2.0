import pandas as pd
import numpy as np
import os

bsc_path = '../../Data/beijing-sentence-corpus/'
info_path = os.path.join(bsc_path, 'BSC.Word.Info.v2.xlsx')
bsc_emd_path = os.path.join(bsc_path, 'BSC.EMD/BSC.EMD.txt')

df = pd.read_excel(info_path, 'word')

wf_bli_column = df['WF_BLI']
len_column = df['LEN']
pred_column = df['PRED']
word_column = df['WORD']
sentence_number_column = df['SN']

modified_word_column = []

for i, word in enumerate(word_column):
    modified_word_column.append(word)

    # Check if the sentence number changes or it's the last row
    if i == len(df) - 1 or sentence_number_column[i] != sentence_number_column[i + 1]:
        # If the current row is the last row or the next row has a different sentence number
        # Append the ampersand symbol to indicate the end of the sentence
        modified_word_column[-1] += '.@'

# Transform the cloze probabilities from logit back to the odds
p = 1 / (1 + np.exp(-2 * df['PRED']))

result_df = pd.DataFrame({'WF_BLI': wf_bli_column, 'LEN': len_column, "PRED": p, "WORD": modified_word_column, "SENT": sentence_number_column})

result_df.to_csv('BSC.txt', sep='\t', index=False)
