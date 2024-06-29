from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd
import torch
import numpy as np
import ast

def calculate_bert_cloze_prob(word, sentence, tokenizer, model):
    """
  Calculates a cloze probability using BERT.

  Args:
      word: The word for which to calculate the cloze probability.
      sentence: The sentence containing the word.
      tokenizer: The BERT tokenizer.
      model: The BERT model.

  Returns:
      A float representing the cloze probability (higher probability indicates the model is more likely to predict the original word).
  """
    if type(sentence) != str and type(word) != str:
        return None  # Return None for NaN sentences

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

        encoding = tokenizer.encode_plus(masked_sentence, return_tensors='pt', add_special_tokens=True, return_attention_mask=True)

        input_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)

        # Get predictions from BERT model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # [1, 29, 30522]

        # Get probability of the original word being predicted
        masked_index = torch.where(encoding['input_ids'] == 103)
        masked_index = masked_index[1].tolist()
        word_index = tokenizer.convert_tokens_to_ids([word])
        cloze_prob = predictions[0, masked_index[0], word_index[0]].item()
        return cloze_prob

    except Exception as e:
        print(f"Error processing row: {word} - {sentence} ({e})")
        return None

def count_present_missing(unique_values, other_list):
  """
  This function counts the number of values present in unique_values that are also present in other_list.

  Args:
      unique_values: A list of unique values.
      other_list: Another list to compare with.

  Returns:
      A tuple containing two counts: (present, missing).
  """

  present = sum(value in other_list for value in unique_values)
  missing = len(unique_values) - present
  return present, missing

def modify_celer_input():
    df = pd.read_csv("CELER.txt", delimiter='\t')
    print(df)  # 574722

    # read the sentence mapping dict
    with open('../../data_splits/CELER_sent_dict.txt', "r") as f:
        data_str = f.read()
        sent_dict = ast.literal_eval(data_str)

    # encode sent ids numerically based on the provided dict from the CELER main training
    df['SENT'] = df['SENT'].map(sent_dict)

    # filter native speakers only (sentences that are used during the training, val and eval of Eyettention model)
    sent_list = list(sent_dict.values())
    df = df[df['SENT'].isin(sent_list)].reset_index(drop=True)
    df['SENT'] = df['SENT'].astype(int)
    print(df)  # 343315

    unique_values = df['SENT'].unique()
    print("unique_values", len(list(unique_values)))  # 4951
    print("sent_list", len(set(sent_list)))  # 5456

    # BERT failed to estimate the cloze prob for some words (e.g., NUM%) --> set them to a value close to 0
    epsilon = 1e-10
    df['PRED'] = df['PRED'].fillna(epsilon)

    # mark the end of the sentence with special symbol @ for the E-Z Reader processing
    modified_word_column = []
    word_column = df['WORD']
    sentence_number_column = df['SENT']
    # Iterate through each row of the DataFrame
    for i, word in enumerate(word_column):
        # Append the word to the modified_word_column
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


def main():
    df = pd.read_csv('/mnt/projekte/pmlcluster/aeye/celer/data_v2.0/data_v2.0/sent_ia.tsv', delimiter='\t')
    df['IA_LABEL'] = df.IA_LABEL.replace('\t(.*)', '', regex=True)

    # Clean up the corpus from None values
    new_df = df[['FREQ_BLLIP', 'WORD_LEN', 'WORD_NORM', 'sentenceid', 'sentence']]
    new_df['word_type'] = new_df['WORD_NORM'].apply(lambda x: type(x).__name__)
    float_rows = new_df[new_df['word_type'] == 'float']
    sentence_ids_to_remove = float_rows['sentenceid'].tolist()
    new_df = new_df.drop(new_df[new_df['sentenceid'].isin(sentence_ids_to_remove)].index)

    word_freq_column = new_df['FREQ_BLLIP']
    len_column = new_df['WORD_LEN']
    word_column = new_df['WORD_NORM']
    sentence_number_column = new_df['sentenceid']

    # Estimate cloze probability using BERT
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    # Store cloze probability estimated by BERT
    new_df['PRED'] = new_df.apply(
        lambda row: calculate_bert_cloze_prob(row['WORD_NORM'], row['sentence'], tokenizer, model), axis=1)

    result_df = pd.DataFrame({'FREQ': word_freq_column, 'LEN': len_column, "PRED": new_df['PRED'], "WORD": word_column,
                          "SENT": sentence_number_column})

    result_df.to_csv('CELER.txt', sep='\t', index=False)
    breakpoint()


if __name__ == "__main__":
   print('going for main')
   main()
   print('main done')
   modify_celer_input()
   print('modify done')

