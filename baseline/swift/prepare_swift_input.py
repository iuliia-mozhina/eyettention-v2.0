import pandas as pd
import numpy as np
from tqdm import tqdm
import ast


def prepare_corpus_file():
    word_info_df = pd.read_csv('../../Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

    # filter sentences that were read only by native speakers
    sub_metadata_path = '../../Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    reader_list = sub_infor[sub_infor.L1 == 'English'].List.values
    sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

    all_sent = pd.DataFrame(columns=["SENT_LEN", "WORD_LEN", "FREQ_BLLIP", "SENT_NUM"])
    for sn_id in tqdm(sn_list):
        # notice: Each sentence is recorded multiple times in file |word_info_df|.
        sn = word_info_df[word_info_df.sentenceid == sn_id]
        sn = sn[sn['list'] == sn.list.values.tolist()[0]]

        sn_str = sn.sentence.iloc[-1]  # the whole sentence as string
        # nessacery sanity check, when split sentence to words, the length of sentence should match the sentence length recorded in celer dataset
        if sn_id in ['1987/w7_019/w7_019.295-3', '1987/w7_036/w7_036.147-43', '1987/w7_091/w7_091.360-6']:
            # extra inverted commas at the end of the sentence
            sn_str = sn_str[:-3] + sn_str[-1:]
        if sn_id == '1987/w7_085/w7_085.200-18':
            sn_str = sn_str[:43] + sn_str[44:]
        sn_len = len(sn_str.split())

        # convert the sentence name to the sentence number
        # read the sentence mapping dict
        with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
            data_str = f.read()
            sent_dict = ast.literal_eval(data_str)
        # encode sent ids numerically based on the provided dict from the CELER main training
        sn['sentenceid'] = sn['sentenceid'].map(sent_dict)

        # Create a DataFrame for the current sentence
        current_sentence_df = pd.DataFrame({
            "SENT_LEN": [sn_len] * len(sn["WORD_LEN"].values),
            "WORD_LEN": sn["WORD_LEN"].values.astype(int),
            "FREQ_BLLIP": sn["FREQ_BLLIP"].values,
            "SENT_NUM": sn["sentenceid"].values.astype(int)
        })
        # Drop rows with NaN values
        current_sentence_df.dropna(inplace=True)
        # Drop rows where WORD_LEN is 0
        current_sentence_df = current_sentence_df[current_sentence_df['WORD_LEN'] != 0]
        # Replace infinity FREQ by 0.0
        current_sentence_df['FREQ_BLLIP'].replace([np.inf, -np.inf], 0.013, inplace=True)
        current_sentence_df['FREQ_BLLIP'] = current_sentence_df['FREQ_BLLIP'].astype(float)

        # Update the sentence length after dropping some columns
        current_sentence_df['SENT_LEN'] = len(current_sentence_df)

        # Append the current sentence data to all_sent
        all_sent = all_sent.append(current_sentence_df, ignore_index=True)

    all_sent.to_csv('corpus_celer.dat', sep='\t', index=False, header=False)


def post_process_swift():
    df = pd.read_csv('SWIFT/SIM/fixseqin_generation.dat', delimiter='\t', header=None)
    df.rename(columns={0: 'sent', 1: 'loc', 2: 'landing_pos', 3: 'dur'}, inplace=True)
    df = df[['sent', 'loc', 'landing_pos', 'dur']]
    df['sent'] = df['sent'] - 1
    df.to_csv('fixseqin_generation_cleaned.csv', sep='\t', index=False)
