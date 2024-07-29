from transformers import BertTokenizer, BertForMaskedLM
import ast
from utils import *


def calculate_bert_cloze_prob(word, sentence, tokenizer, model):
    try:
        # convert the first word of a sentence to lowercase
        if sentence[0].isupper():
            rest_of_sentence = sentence[1:]
            sentence = sentence[0].lower() + rest_of_sentence

        # Mask the word in the sentence
        if sentence.startswith(word + ' '):
            masked_sentence = f"[MASK] {sentence[len(word) + 1:]}"
        else:
            parts = sentence.split(word, 1)
            if len(parts) == 1:
                return None
            masked_sentence = f"{parts[0]} [MASK] {parts[1]}"

        encoding = tokenizer.encode_plus(masked_sentence, return_tensors='pt', add_special_tokens=True,
                                         return_attention_mask=True)

        input_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)

        # Get predictions from BERT model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get probability of the original word being predicted
        masked_index = torch.where(encoding['input_ids'] == 103)
        masked_index = masked_index[1].tolist()
        word_index = tokenizer.convert_tokens_to_ids([word])
        cloze_prob = predictions[0, masked_index[0], word_index[0]].item()
        return cloze_prob

    except Exception as e:
        print(f"Error processing row: {word} - {sentence} ({e})")
        return None


def modify_celer_input():
    df = pd.read_csv("CELER.txt", delimiter='\t')

    # BERT failed to estimate the cloze prob for some words (e.g., NUM%) --> set them to a value close to 0
    epsilon = 1e-5
    df['PRED'] = df['PRED'].fillna(epsilon)

    # mark the end of the sentence with special symbol @ for the E-Z Reader processing
    modified_word_column = []
    word_column = df['WORD']
    sentence_number_column = df['SENT']
    for i, word in enumerate(word_column):
        modified_word_column.append(word)
        # Check if the sentence number changes or it's the last row
        if i == len(df) - 1 or (i < len(df) - 1 and sentence_number_column[i] != sentence_number_column[i + 1]):
            # If the current row is the last row or the next row has a different sentence number
            # Append the ampersand symbol to indicate the end of the sentence
            modified_word_column[-1] += '.@'
    df['WORD'] = modified_word_column

    # revert the negative log transformation with base 2 for word frequency
    df['FREQ'] = np.power(2, -df['FREQ'])
    # convert word length to int
    df['LEN'] = df['LEN'].astype(int)
    # remove ' signs from words (e.g., don't --> dont)
    df['WORD'] = df['WORD'].str.replace("'", '')

    df.to_csv('CELER_input.txt', sep='\t', index=False)


def calculate_cloze_prob():
    df = pd.read_csv("CELER_BERT_input.txt", delimiter='\t')

    word_freq_column = df['FREQ_BLLIP']
    len_column = df['WORD_LEN']
    word_column = df['WORD_NORM']
    sentence_number_column = df['SENT']

    # Estimate cloze probability using BERT
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    # Store cloze probability estimated by BERT
    df['PRED'] = df.apply(
        lambda row: calculate_bert_cloze_prob(row['WORD_NORM'], row['sentence'], tokenizer, model), axis=1)

    result_df = pd.DataFrame({'FREQ': word_freq_column, 'LEN': len_column, "PRED": df['PRED'], "WORD": word_column,
                              "SENT": sentence_number_column})
    result_df.to_csv('CELER.txt', sep='\t', index=False)


def preprocess_celer_input():
    word_info_df = pd.read_csv('../../Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
    word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

    # filter sentences that were read only by native speakers
    sub_metadata_path = '../../Data/celer/metadata.tsv'
    sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
    reader_list = sub_infor[sub_infor.L1 == 'English'].List.values
    sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

    all_sent = pd.DataFrame(columns=list(word_info_df.columns))
    for sn_id in tqdm(sn_list):
        # notice: Each sentence is recorded multiple times in file |word_info_df|.
        sn = word_info_df[word_info_df.sentenceid == sn_id]
        sn = sn[sn['list'] == sn.list.values.tolist()[0]]
        all_sent = all_sent.append(sn, ignore_index=True)

    # read the sentence mapping dict
    with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
        data_str = f.read()
        sent_dict = ast.literal_eval(data_str)
    # encode sent ids numerically based on the provided dict from the CELER main training
    all_sent['SENT'] = all_sent['sentenceid'].map(sent_dict)

    df = all_sent[['FREQ_BLLIP', 'WORD_LEN', 'WORD_NORM', 'SENT', 'sentence']]
    df = df.dropna(subset=['WORD_NORM'])
    df.to_csv('CELER_BERT_input.txt', sep='\t', index=False)


if __name__ == "__main__":
    preprocess_celer_input()
    calculate_cloze_prob()
    modify_celer_input()
