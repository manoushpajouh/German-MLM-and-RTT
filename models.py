#!/usr/bin/env python3

# Author: Manoush Pajouh 
# Project Name: Masked Token Predictions Differences for German and RTT Models
# Course: CMPU-366
# Due Date: December 14, 2025

# GERMAN ONLY MODEL (USING GERMAN-BERT)----------------------------------------

# IMPORTS ---------------------------------------------------------------------
# Use to finetune and use the German model onto the German texts 
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, AutoModelForSeq2SeqLM, DataCollatorForLanguageModeling
# Importing path for easier accessibility
from pathlib import Path
# Used to shuffle data 
import random
# Used for perplexity 
import math
# Used for removing punctuation/data processing
import string
# Used for evaluation method 
import torch
# used for the translation function 
from typing import List


# DATA PROCESSING -------------------------------------------------------------
# Prepare the texts
# .de/.en file is one sentence per line
def prep_data(file: str | Path) -> list[str]:
    """Prepare the dataset by taking in a file and processing it into a list
    of strings""" 
    # Open and load the sentences from the file 
    with open(file, "r", encoding="utf-8") as f:
        # Make sure to split the lines 
        sentences = f.read().splitlines()
    return sentences

# Filter out sentence that are less than 3 words long. 
def filter_short_sents(sentences:list[str]) -> list[str]:
    """ Takes in a list of sentences and returns the same list of sentences
    with any sentences less than 3 words long removed."""

    # Initialize list of sentences to keep
    long_sents = []
    # Iterate through all inputted sentences 
    for sentence in sentences:
        # Strip each sentence of whitespace
        sentence = sentence.strip()
        # Remove trailing punctuation (prevents predicting period/punctuation)
        sentence = sentence.rstrip(string.punctuation)
        # Split the sentences into tokens
        tokens = sentence.split()
        # If more than three tokens, append to long_sents 
        if len(tokens) >= 3:
            # Using join with a space so that all tokens are restructured as a sentence
            long_sents.append(" ".join(tokens))
    # Return long sentences
    return long_sents


# Need to then split the only German dataset into 80/20 for training/testing
# Hard-coding the training percentage to be 0.8 but can be modified
def split_train_test(sentences: list[str], train_percentage: 0.8) -> tuple[list[str], list[str]]:
    """Takes in a list of strings and returns the same list split into two parts;
    these are the training and testing percentages. Train_percentage
    must be out of 1, and testing percentage is 1-train_percentage. Ex. if 
    train_percentage is 0.8 then the list is cut into one list of 80% of items 
    (training) and another list with 20% of the items (testing)."""
    # Find where in the list to split 
    # Make sure to round to int
    index = int(len(sentences) * train_percentage)
    # # Manually setting the seed - can pick any number - used for consistency
    # random.seed(42)
    # # Randomly shuffle the list for no particular order
    # random.shuffle(sentences)
    # Separate the two lists: training which has from beginning to index (80%)
    training = sentences[:index]
    # and the rest is testing (from index to end (20%))
    testing = sentences[index:]
    # Return the two lists
    return training, testing

# MASKING FUNCTION -----------------------------------------------------
def mask_dataset(sentences: list[str], tokenizer):
    """Masks one random single-token word per sentence.
    Returns a list of dicts with masked_sentence and true_token_id."""

    # Initialize list of sentences with the mask applied
    masked = []
    
    # For all sentences in the dataset 
    for sentence in sentences:
        # Strip sentence of punctuation and leading/trailing whitespace
        sentence = sentence.strip().rstrip(string.punctuation)
        # Split the sentence into a list of words/tokens
        words = sentence.split()

        # Pick a random index in the list of words of the sentence to mask
        mask_index = random.randrange(len(words))
        # Extract the word that will be masked
        true_word = words[mask_index]

        # Tokenize true word
        true_ids = tokenizer(true_word, add_special_tokens=False)["input_ids"]

        # Skip multi-token words
        if len(true_ids) != 1:
            continue

        # Extract the id of the true word
        true_token_id = true_ids[0]

        # Duplicate the list of words in the sentence
        masked_words = words.copy()

        # Replace the word to be masked with the masking token
        # In this case, the masking token is [MASK]
        masked_words[mask_index] = tokenizer.mask_token

        # Add all of the entries into the dataset manually
        masked.append({
            # Rebuild sentence with masked token
            "masked_sentence": " ".join(masked_words),
            # Add true word to the dictionary 
            "true_word": true_word,
            # Add the token id to the dictionary
            "true_token_id": true_ids[0]
        })

    # Return the dict
    return masked

# IMPORT MODELS AND TOKENIZERS FOR TRANSLATION FUNCTIONS ----------------------
# Define the German --> English path name
ger_eng_path = "Helsinki-NLP/opus-mt-de-en"
# Define the German --> English tokenizer
ger_to_eng_tokenizer = AutoTokenizer.from_pretrained(ger_eng_path)
# Define the German --> English model
ger_to_eng_model = AutoModelForSeq2SeqLM.from_pretrained(ger_eng_path)

# Define English --> German path name
eng_ger_path = "Helsinki-NLP/opus-mt-en-de"
# Define English --> German tokenizer
eng_to_ger_tokenizer = AutoTokenizer.from_pretrained(eng_ger_path)
# Define English --> German model
eng_to_ger_model = AutoModelForSeq2SeqLM.from_pretrained(eng_ger_path)

# TRANSLATION FUNCTION FOR RTT ---------------------------------------------

def translate_masked(masked_dataset: list[dict], model, tokenizer, batch_size: int = 16, placeholder: str = "ZZZMASKZZZ") -> list[dict]:
    """Translate masked sentences to English while preserving [MASK] token.
    Using a temporary placeholder for [MASK] to prevent it from being translated.
    Model and tokenizer are that of the translation model. Translating
    in batches for efficiency."""

    # Initialize dataset to return
    translated_dataset = []

    # Iterate through the entire length of dataset and put into batches
    for i in range(0, len(masked_dataset), batch_size):
        # Splice the list and create batches from i to i + batch_size
        batch = masked_dataset[i : i + batch_size]

        # For all sentences in the batch, replace [MASK] from bert
        # to ZZZMASKZZZ to prevent it from being overwritten by translator model
        masked_sentences = [pair["masked_sentence"].replace("[MASK]", placeholder) for pair in batch]

        # Tokenize the entire batch of sentences
        tokenized = tokenizer(masked_sentences, return_tensors="pt", padding=True, truncation=True)

        # Disable gradient calculation
        with torch.no_grad():
            # Translate the entire sentence using the translator model
            outputs = model.generate(**tokenized, max_length=128)

        # Decode the output and convert to human readable respnse
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Restore [MASK] and rebuild dataset
        # For each pairing and translated text in the batch
        for pair, translated_text in zip(batch, translated):
            # Add the pairing and the sentence it translated to the dictionary
            translated_dataset.append({
                # Used to convert to english sentences
                # Restore ZZZMASKZZZ to [MASK] so it can be used for BERT model
                "masked_sentence_en": translated_text.replace(placeholder, "[MASK]"),
                # Adding the true word into the dataset
                # Neglecting the id since we do not need it for rtt
                "true_word": pair["true_word"]
            })

    # Return the translated dataset with the [MASK] token in each sentence
    return translated_dataset


def english_to_german(sentence: str) -> str:
    """Input an English sentence and return the German translation using
    the Helsinki-NLP model."""
    # Tokenize the sentence
    tokens = eng_to_ger_tokenizer(sentence, return_tensors="pt", truncation=True)
    # Disable gradient descent
    with torch.no_grad():
        # Generate output prediction
        outputs = eng_to_ger_model.generate(input_ids=tokens["input_ids"], max_new_tokens=128)
    # Decode output to put in human readable form
    translated = eng_to_ger_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return translated
    return translated


# PREDICTION FUNCTION ----------------------------------------------------
def predict_top_k(model, tokenizer, masked_sentence: str, k: int) -> list:
    """Returns a list of the top k words that can be used to replace the 
    masking token"""

    # Tokenize the sentence with the tokenizer (turns to tensors)
    tokenized = tokenizer(masked_sentence, return_tensors="pt")

    # Disable gradient descent
    with torch.no_grad():
        # Feed that into the model and generate the output prediction for [MASK]
        outputs = model(**tokenized)

    # Extract all logits for all tokens
    logits = outputs.logits[0]
    # Identify token index of [MASK] within the sentence
    mask_token_index = (tokenized["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

    # Calculate probaility using softmax function
    probs = torch.softmax(logits[mask_token_index], dim=-1)

    # Create list of top k predictions
    predictions = torch.topk(probs, k).indices.tolist()

    # Return the predictions 
    return predictions


# EVALUATION METHODS -----------------------------------------
def top_k_accuracy_ger(model, tokenizer, masked_dataset: list[dict], k: int) -> float:
    """Takes in a model, tokenizer, masked_dataset and k in order
    to evaluate the model's accuracy over the masked_dataset, which it then returns. 
    Function calculates to however many k is inputted. Compares token ids
    to ensure functionality for k=1 as well as higher numbers."""
    # Initialize correct number of predictions
    correct = 0
    # Initialize total predictions made
    total = 0

    # Iterate through all pairs of ids and sentences in the masked dataset
    for pair in masked_dataset:
        # Calculate the list of the top k predictions
        top_k_predictions = predict_top_k(model, tokenizer, pair["masked_sentence"], k)
        
        # If the id is in the list of predictions
        if pair["true_token_id"] in top_k_predictions:
            # Increment the correct score
            correct += 1
        
        # Regardless of correctness, increment total sentences processed
        total += 1

    # Calculate the accuracy
    accuracy = correct / total
    # Return the accuracy 
    return accuracy

def top_k_accuracy_eng(model, tokenizer, masked_dataset: list[dict], k: int) -> float:
    """Takes in a model, tokenizer, masked_dataset (english version) and k in 
    order to evaluate the model's accuracy over the masked_dataset, which it 
    then returns. Calls prediction function to the same k, translates each
    of the predictions, and then runs the accuracy model. Compares 
    token ids to ensure functionality for k=1 as well as higher numbers. """

    # Initialize correct number of predictions
    correct = 0
    # Initialize total predictions made
    total = 0

    # Iterate through all pairs of ids and sentences in the masked dataset
    for pair in masked_dataset:
        # Some of the sentences may not have the mask in it anymore after translation
        if tokenizer.mask_token not in pair["masked_sentence_en"]:
            # If mask is not in sentence anymore, skip that sentence
            continue

        # Calculate the list of the top k predictions for each sentence
        top_k_predictions = predict_top_k(model, tokenizer, pair["masked_sentence_en"], k)

        # Initialize the list of German predictions
        top_k_predictions_ger = []
        # Extract the true word 
        true_word = pair["true_word"]
        # For each English prediction
        for eng_id in top_k_predictions:
            # Isolate the english token
            eng_token = tokenizer.decode(eng_id, skip_special_tokens=True).strip()
            # Skip if none
            if not eng_token:
                continue

            # Put the word back into the english sentence replacing "MASK"
            filled_sentence = pair["masked_sentence_en"].replace("[MASK]", eng_token) 

            # Give the whole sentence to the translator
            filled_sentence_ger = english_to_german(filled_sentence)

            # If the true word from German is in the translated ver of the sentence
            if true_word in filled_sentence_ger:
                # Increment correct counter
                correct += 1
                break

        # Increment the counter for the total number of processed sentences
        total += 1

    # Return accuracy
    return correct / total


# MAIN ----------------------------------------------------------------------

def main():
    # Prepare the German dataset by reading the file and returning a list of sentences
    german_sentences = prep_data("datasets/open_subtitles/OpenSubtitles.de-en.de")
    
    # Using a smaller subset of the data for faster train/test (20% of the og 70k)
    subset = int(len(german_sentences) * 0.20) 
    german_sentences = german_sentences[:subset]

    # Filter out sentences that are too short 
    german_sentences = filter_short_sents(german_sentences)
    # Split the sentences into training and testing 
    german_training, german_testing = split_train_test(german_sentences, 0.8)

    # Convert the lists of sentences into HuggingFace DatasetDicts to use
    # HuggingFace expects the words 'train' and 'test'
    german_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": german_training}),
        "test": Dataset.from_dict({"text": german_testing})
    })

    # Initialize the model and tokenizer
    # Isolate name of model from Hugging Face 
    german_gpt_name = "google-bert/bert-base-german-cased"

    # Initialize the German tokenizer 
    german_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-german-cased")
    # Tokenizer needs eos token as padding (HF)
    # german_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if german_tokenizer.pad_token is None:
        german_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
 
    # Pre-defined HuggingFace function needed for tokenization 
    def german_tokenize(batch):
        return german_tokenizer(batch["text"], truncation=True, max_length=1024)

    # Use loaded tokenizer for the German dataset
    # Use the map function so it does it for all text values aka sentences
    # This is the HF .map specific to Datasets not the python map()
    tokenized_german = german_dataset.map(german_tokenize, batched=True)

    # Initialize the model BERT German version
    # german_bert_model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-german-cased")
    # If you are re-evaluating and have already trained the model --> load from checkpoint
    checkpoint_path = "german_bert_model/checkpoint-2250"
    german_bert_model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)

    # Need to resize the token embeddings for fine-tuning process
    # This way new tokens are added 
    german_bert_model.resize_token_embeddings(len(german_tokenizer))

    # MLM expects data collator
    data_collator = DataCollatorForLanguageModeling(
        # Feed in tokenizer
        tokenizer=german_tokenizer,
        # Masked language modeling set to true
        mlm=True,
        # Probability is 15% by standard
        mlm_probability=0.15
    )

    # Initialize training arguments
    args = TrainingArguments(
        # Save the model into a folder named german_bert_model
        output_dir="german_bert_model",
        # Each time you run the model, overwrite anything in the folder already 
        overwrite_output_dir=True,
        # Set batch size for training
        per_device_train_batch_size=4,
        # Setting more steps to make sure that simulates 12 since batch is small (3)
        gradient_accumulation_steps=3,
        # Set training epochs
        num_train_epochs=3,
        # Set learning rate
        learning_rate=5e-5,
        # Set weight decay
        weight_decay=0.01,
        # Learning rate is gradually increased (stabilizes traning)
        warmup_steps=100,
        # Log steps in order to display progress
        logging_steps=50,
        # Save steps for model 
        save_steps=500,
    )

    # Initialize the trainer with the model, args, dataset (train/test), and data collator 
    trainer = Trainer(model=german_bert_model, args=args, 
                    # Make sure to set everything to the values in train/test
                    train_dataset=tokenized_german["train"], 
                    eval_dataset=tokenized_german["test"], 
                    # Specify the data collator
                    data_collator=data_collator,
    )

    # Train the model 
    # trainer.train() 

    # EVALUATION OF GERMAN MODEL ----------------------------------------------

    # Evaluate the model with eval mode
    german_bert_model.eval() 

    # # Mask the testing sentences
    masked_german = mask_dataset(german_testing, german_tokenizer)

    # Calculate accuracy scores using the german masked testing sentences
    top1 = top_k_accuracy_ger(german_bert_model, german_tokenizer, masked_german, k=1)
    top5 = top_k_accuracy_ger(german_bert_model, german_tokenizer, masked_german, k=5)
    top10 = top_k_accuracy_ger(german_bert_model, german_tokenizer, masked_german, k=10)

    # print accuracy scores
    print(f"Top-1 Masked Accuracy: {top1:.4f}")
    print(f"Top-5 Masked Accuracy: {top5:.4f}")
    print(f"Top-10 Masked Accuracy: {top10:.4f}")

    # RTT MODEL -------------------------------------------------------------
    # Prepare the English dataset by reding the file and returning list of sentences
    english_sentences = prep_data("datasets/open_subtitles/OpenSubtitles.de-en.en")
    # Do the same split for the English sentences to maintain alignment
    # Using the same subset and measurements without shuffling so all sentences
    # align at least in content
    english_sentences = english_sentences[:subset]
    # Split English sentences into training and testing
    english_training, english_testing = split_train_test(english_sentences, 0.8)

    # Convert the lists of sentences into HuggingFace DatasetDicts to use
    # Using the word 'text' since HF Datasets need column format so it expects it
    # Doesn't have to be the word text, just using it for consistency 
    english_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": english_training}),
        "test": Dataset.from_dict({"text": english_testing})
    })

    # Initialize the model BERT English version
    # english_bert_model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-cased")
    # If you are re-evaluating and have already trained the model --> load from checkpoint
    eng_checkpoint_path = "english_bert_model/checkpoint-2823"
    english_bert_model = AutoModelForMaskedLM.from_pretrained(eng_checkpoint_path)

    # Initialize the BERT English tokenizer
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    # Pre-defined HuggingFace function needed for tokenization 
    # Batch tokenize (separate from German tokenizer/tokenize())
    def english_tokenize(batch):
        return english_tokenizer(batch["text"], truncation=True, max_length=1024)

    # Use loaded tokenizer for the English dataset
    # Use the map function so it does it for all text values aka sentences
    # Set remove_columns so the key 'text' is not looked at
    # This is the HF .map specific to Datasets not the python map()
    tokenized_english = english_dataset.map(english_tokenize, batched=True)

    # Need to resize the token embeddings for fine-tuning process
    # This way new tokens are added 
    english_bert_model.resize_token_embeddings(len(english_tokenizer))

    # MLM expects data collator
    data_collator = DataCollatorForLanguageModeling(
        # Feed in tokenizer
        tokenizer=english_tokenizer,
        # Masked language modeling set to true
        mlm=True,
        # Probability is 15% by standard
        mlm_probability=0.15
    )

    # Initialize training arguments --> all the same as the German trainer
    args = TrainingArguments(
        # Save the model into a folder named german_bert_model
        output_dir="english_bert_model",
        # Each time you run the model, overwrite anything in the folder already 
        overwrite_output_dir=True,
        # Set batch size for training
        per_device_train_batch_size=4,
        # Setting more steps to make sure that simulates 12 since batch is small (3)
        gradient_accumulation_steps=3,
        # Set training epochs
        num_train_epochs=3,
        # Set learning rate
        learning_rate=5e-5,
        # Set weight decay
        weight_decay=0.01,
        # Learning rate is gradually increased (stabilizes traning)
        warmup_steps=100,
        # Log steps in order to display progress
        logging_steps=50,
        # Save steps for model 
        save_steps=500,
    )

    # Initialize the trainer with the model, args, dataset (train/test), and data collator 
    trainer = Trainer(model=english_bert_model, args=args, 
                    # Make sure to set everything to the values in train/test
                    train_dataset=tokenized_english["train"], 
                    eval_dataset=tokenized_english["test"], 
                    # Specify the data collator
                    data_collator=data_collator,
    )

    # Train the model
    # trainer.train() 

    # EVALUATION OF ENGLISH MODEL ----------------------------------------------

    # Evaluate the model with eval mode
    english_bert_model.eval() 

    # Convert the masked German sentences to masked English sentences
    # Give it the translation model and the tokenizer of that model
    # Keep the batch size as 16
    masked_english = translate_masked(masked_german, ger_to_eng_model, ger_to_eng_tokenizer, 16)

    # # Calculate accuracy scores using the English masked testing sentences 
    top1 = top_k_accuracy_eng(english_bert_model, english_tokenizer, masked_english, k=1)
    top5 = top_k_accuracy_eng(english_bert_model, english_tokenizer, masked_english, k=5)
    top10 = top_k_accuracy_eng(english_bert_model, english_tokenizer, masked_english, k=10)

    # Print accuracy scores
    print(f"Top-1 Masked Accuracy: {top1:.4f}")
    print(f"Top-5 Masked Accuracy: {top5:.4f}")
    print(f"Top-10 Masked Accuracy: {top10:.4f}")


if __name__ == "__main__":
    main()
    
