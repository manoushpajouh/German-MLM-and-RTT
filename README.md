# Masked Word Prediction Differences for German-English RTT Models and German Models

## Description
This project runs two models and compares them using Top-k accuracy scores. One of 
the models is a simple masked language model using the OPUS data (see Dataset
section for more details).The MLM is trained and tested on purely German data. 
These results are then compared to that of the second model, which uses round-trip
translation to predict the same masked token. To do so, it takes in a German 
sentence with the mask already in place, translates the entire sentence to 
English and retaining the masking token, predicts the masked word, and then 
translates the entire sentence (with the word prediction replacing the masking
token) back into German to compute results. Ultimately, these models are compared
alongside each other to investigate using RTT as a masked language model for
languages with morphological differences.

The translations of the sentences are implemented using the Helsinki-NLP Marian
Neural Machine Translation Models, namely German to English and English to German. 


## Dataset 
This program uses the OPUS Open Subtitles v1 corpus. I have used a subset of 
approximately 20% for each model in order to condense the corpora for efficiency 
purposes. This subset is then split between 80% for training and 20% for testing.

The German dataset is used for both models, whereas the English
dataset is used only for the RTT model. 

Once downloaded, the datasets need to be processed. They are formatted to be
one subtitle per line, with the first line in the German document (.de) aligning
with the first subtitle in the English document (.en), and so on and so forth. 

## Models 
I am using two google-bert models available on Hugging Face. The first is designed
for [English](https://huggingface.co/google-bert/bert-base-cased) sentences. The other
is also intended for masked language modeling and is designed for 
[German](https://huggingface.co/google-bert/bert-base-german-cased)
sentences.

As mentioned in Section "Description", I am using the Helsinki-NLP Marian Neural 
Machine Translation models available on HuggingFace for translations; 
Helsinki-NLP/opus-mt-de-en has been used for [German to English translation](https://huggingface.co/Helsinki-NLP/opus-mt-de-en)
and Helsinki-NLP/opus-mt-en-de for [English to German translation](https://huggingface.co/Helsinki-NLP/opus-mt-en-de).
These are available at 




## Features
* Trained and tested BERT models
* Application of masks onto tokens within sentences
* Translating German --> English --> German 
* Predicted masked words
* Top-k Accuracy score calculations

## Installation/Setup

Begin by installing the required dependencies. You may need to adjust by writing
pip3 instead. 

``` pip install -r requirements.txt ```

``` python3 -m spacy download en_core_web_sm ```

If you are using a virtual environment:

Set up the virtual environment. 

``` python3 -m venv venv ```

Activate it. 

``` source venv/bin/activate ```

Install the necessary requirements.

``` pip install --upgrade pip ```

``` pip install -r requirements.txt ```

``` pip install torch --index-url https://download.pytorch.org/whl/cpu ```



## Usage
To run the file, open a terminal and make sure to be in the directory that
houses the 'models.py' file. After making sure you have all the necessary 
requirements, to run the file 
containing the models, just run python on the models.py file.

``` python3 models.py ```

The code contains both the German MLM and the RTT model. These can be
trained and evaluated at the same time, but it may take a long time. I recommend
commenting out portions of .main() specified to be for either model, and 
isolating them. You have to run ```trainer.train()``` before evaluation and make 
sure that the model is initialized from a url. Comment out model initialization
from a checkpoint. 

However, you only have to do this once, since checkpoints are loaded onto 
folders in the same directory as ```models.py```. This means that you can then
comment out ```trainer.train()``` and comment out the model initialization
from the url. Instead, uncomment loading the model from a checkpoint for faster
results since the model does not need to be trained again. 

## Results
The German MLM had an accuracy rate of 42.68%,
a top-5 accuracy rate of 66.63% and a top-10 ac-
curacy rate of 74.16%. These results showcase
that the German BERT model can predict masked
tokens well, and substantially improves as the num-
ber of accepted predictions increases. The RTT
model had an accuracy rate of 14.29%, a top-5
accuracy rate of 28.57%, and a top-10 accuracy
rate of 42.86%. The German MLM is by far more
successful consistently. The RTT model’s accuracy
scores improve as k increases, but the overall scores
are much poorer than those of the German MLM.
This gap in performance highlights the difficulty
and noise when introducing cross-lingual transfer
and back-translation. This is mainly because errors
accumulate across the translation and prediction
stages. This suggests that correct predictions are
often among lower-ranked options but fail to appear
as the single most-probable choice.

## Biases and Limitations

* Corpus: This model is limited by the nature of translated cor-
pora as a whole, meaning that there is bias and in-
consistency between the two versions of the dataset
– English and German – that the models have been
trained on. Furthermore, the dataset is limited to
movie subtitles, so it will not be neither extremely
formal nor informal. Just as it may not represent
natural human language and slang entirely accu-
rately, it will not represent formal vocabulary, such
as medical or legal. As with any translated set,
but especially for subtitles which prioritize captur-
ing tone and emotion, more or less context may
be necessary within each translation. Because of
this, translations will not be direct, word-for-word
translations.
Additionally, for faster training and testing times,
I use a subset of the dataset: this subset is limited to
the first 20% of the sentences as they appear rather
in random order. I also filter out sentences that
contain only one word to give the model context
within the sentence. This filtering introduces some
bias, since the sentences are not randomly shuffled
on my part, despite the sentences not being in any
particular order.

* BERT Models:
Any biases present in the models incorporated into
this study are naturally going to have the same bias
for this case study.

* Translation Models
In the RTT model, by translating sentences with
the masked token and then with the filled-in pre-
diction, there is a chance that the translation model
is changing the sentence rather than reverting it to
the exact German sentence. This is especially a
concern since BERT models treat prefix and suffix
tokens as individual word tokens. Furthermore, I
filter out sentences that, once the mask has been
applied and the sentence is translated back to Ger-
man, no longer have the mask in the sentence due
to translation errors. This could be because the
sentence no longer needs the mask to make sense
(i.e. "Ich habe den Film gesehen" → "Ich habe den
[MASK] gesehen" → "Ich habe den gesehen").

* Evaluation:
Due to the translation issues in which some sen-
tences may not contain the masking token at all, I
have filtered out these sentences in the evaluation
stage. This, however, means that the German and
English versions of the models are not being tested
on exactly the same sentences in terms (since the
RTT model translates them and some data may be
noisy), though they are consistent in content, and
not the same quantity of sentences (since there is
more filtering done). This may be a contributing
factor as to why the accuracy rates increase at a
proportional rate.

# Links and Resources

OPUS Open Subtitles German-English: https://opus.nlpl.eu/OpenSubtitles/en&de/v2024/OpenSubtitles

BERT Models:

* German: https://huggingface.co/google-bert/bert-base-german-cased 

* English: https://huggingface.co/google-bert/bert-base-cased

Helsinki Models:

* German to English: https://huggingface.co/Helsinki-NLP/opus-mt-de-en

* English to German: https://huggingface.co/Helsinki-NLP/opus-mt-en-de 

## Paper and More Information
Please see the file containing the scientific paper for full details on 
the project (training, testing, masking, etc.) and all references. 

## Authors and Contacts 
Manoush Pajouh (Vassar College)

GitHub: https://github.com/manoushpajouh  