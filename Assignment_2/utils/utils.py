from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import skseq
import skseq.sequences
import skseq.readers
from skseq.sequences import sequence
from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary
from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.extended_feature import ExtendedFeatures
from skseq.sequences.structured_perceptron import StructuredPerceptron
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report


from IPython.display import display, HTML

############################################################################################################
# No need to have this function since we do have tiny_dataset.csv provided!
def tiny_test():
    """
    Creates a tiny test dataset.

    Returns:
      A tuple (X, y) for each word/special character in a sentence X, the list of tags correspoding to it. 
    """
    X = [['The programmers from Barcelona might write a sentence without a spell checker . '],
         ['The programmers from Barchelona cannot write a sentence without a spell checker . '],
         ['Jack London went to Parris . '],
         ['Jack London went to Paris . '],
         ['Bill gates and Steve jobs never though Microsoft would become such a big company . '],
         ['Bill Gates and Steve Jobs never though Microsof would become such a big company . '],
         ['The president of U.S.A though they could win the war . '],
         ['The president of the United States of America though they could win the war . '],
         ['The king of Saudi Arabia wanted total control . '],
         ['Robin does not want to go to Saudi Arabia . '],
         ['Apple is a great company . '],
         ['I really love apples and oranges . '],
         ['Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York . ']]

    y = [['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'I-geo', 'I-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O'],
            ['B-org', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'B-per', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo',
             'I-geo', 'O']]

    return [i[0].split() for i in X], y
############################################################################################################
def f1_score_weighted(y_true, y_pred):
    """
    Computes the weighted F1 scoreCompute the F1 score, 
    also known as balanced F-score or F-measure, 
    based on the y_true and y_pred.

    Returns:
      The weighted F1 score.
    """
    return f1_score(y_true, y_pred, average='weighted')
############################################################################################################
def accuracy(y_true, y_pred):
    """
    Calculate accuracy on the set but taking into account only those predictions where the ground truth is not "O".

    Parameters:
    y_true (list): The list of true labels.
    y_pred (list): The list of predicted labels.

    Returns:
    float: The accuracy score, representing the proportion of correct predictions.
    """
    # Filter out 'O' labels and corresponding predictions
    filtered_pairs = [(true, pred) for true, pred in zip(y_true, y_pred) if true != 'O']
    
    # Separate the filtered true and predicted labels
    filtered_y_true, filtered_y_pred = zip(*filtered_pairs) if filtered_pairs else ([], [])

    # Calculate and return the accuracy score
    return accuracy_score(filtered_y_true, filtered_y_pred) if filtered_y_true else 0.0




############################################################################################################
def plot_confusion_matrix(y_true, y_pred, dict_tag):
    """
    Plots a confusion matrix using a heatmap.

    Parameters:
      y_true (list or array): True labels.
      y_pred (list or array): Predicted labels.
      dict_tag (dict): Dictionary mapping tag values to their corresponding labels.

    Returns:
      Confusion matrix plot
    """
    reverse_tag_dict = {v: k for k, v in dict_tag.items()}

    # Ensure all unique tags are included
    unique_tags = np.unique(np.concatenate((y_true, y_pred)))

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_tags)

    # Map the integer tags to string labels for DataFrame
    index_labels = [reverse_tag_dict.get(tag, tag) for tag in unique_tags]
    columns_labels = [reverse_tag_dict.get(tag, tag) for tag in unique_tags]

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=index_labels, columns=columns_labels)

    # Plotting the confusion matrix
    fig_size = max(10, 0.8*len(index_labels) / 2), 0.8*max(10, len(columns_labels) / 2)

    # Plotting the confusion matrix
    plt.figure(figsize=fig_size)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap="rocket_r",  cbar=False, xticklabels = True)
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.title('Confusion Matrix')
    plt.show()
########################################################################################
def transform_data_sentence_tag(df):
  """
  Extracts sentences and tags from the provided df.

  Parameters:
      df: A df object containing sentence_id, words, tags columns.

  Returns:
      A tuple (X, y) containing sentences and corresponding tags .
  """
  # Sentence, Tag
  X, y = [], []

  # Unique ids
  id_s = df.sentence_id.unique()  

  # Progress bar using tqdm library
  progress = tqdm(id_s, desc="Creating", unit= "sentence" )

  for sentence in progress: 
      # Append the words for the current sentence as list of X
      X.append(list(df[df["sentence_id"] == sentence]["words"].values))
      # Append the tags of all values in X list
      y.append(list(df[df["sentence_id"] == sentence]["tags"].values))

  return X, y
############################################################################################################
def create_corpus(sentences, tags):
    """
    Create a corpus from a list of sentences and corresponding tags.

    Parameters:
        sentences (list): A list of sentences.
        tags (list): A list of corresponding tags for each sentence.

    Returns:
        word_dict (dict): A dictionary mapping words to integer indices.
        tag_dict (dict): A dictionary mapping tags to integer indices.
    """
    # Initialize word and tag dictionaries
    word_dict = {}
    tag_dict = {}

    # Iterate over sentences and tags to build the corpus and dictionaries
    for sentence, tag_sequence in zip(sentences, tags):
        word_indices = []
        tag_indices = []
        
        for word, tag in zip(sentence, tag_sequence):
            # Add word to word dictionary if not present
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            # Add tag to tag dictionary if not present
            if tag not in tag_dict:
                tag_dict[tag] = len(tag_dict)
            
            # Append word and tag indices to the current sequence
            word_indices.append(word_dict[word])
            tag_indices.append(tag_dict[tag])

    return word_dict, tag_dict
############################################################################################################
def create_sequence_list(X, y, word_dict, tag_dict):
    """
    Create a SequenceList object from training data.

    Parameters:
        X (list): A list of sentences.
        y (list): A list of corresponding tags for each sentence.
        word_dict (dict): A dictionary mapping words to integer indices.
        tag_dict (dict): A dictionary mapping tags to integer indices.

    Returns:
        SequenceList: A SequenceList object containing the sequences.
    """
    # Create a SequenceList object using the word and tag dictionaries
    sequence_list = SequenceList(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    # Iterate over the sentences and tags to add sequences to the SequenceList
    for sent_x, sent_y in tqdm(zip(X, y), desc="Creating Sequence List", total=len(X)):
        sequence_list.add_sequence(sent_x, sent_y, LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return sequence_list


def load_p(path):
    """
    Load data from a pickle file.

    Parameters:
        path (str): The path to the pickle file.

    Returns:
        object: The loaded data.
    """
    with open(path, 'rb') as file:
        df_ = pickle.load(file)
    return df_

def save_p(path, data):
    """
    Save data to a pickle file.

    Parameters:
        path (str): The path to save the pickle file.
        data (object): The data to be saved.
    """
    with open("fitted_models\\"+path, "wb") as f:
        pickle.dump(data, f)



def get_accuracy(model, data, true_tags_dict, corpus_tag_dict):
    """
    Calculate accuracy metrics for a given model and dataset.

    Parameters:
        model (object): The trained model object.
        data (list): A list of sentences.
        true_tags_dict (dict): A dictionary containing the true tags for each sentence.
        corpus_tag_dict (dict): A dictionary mapping tags to indices.

    Returns:
        tuple: A tuple containing the predicted tags list, the number of correct sentences, the total number of correct tags, and the total number of tags.
    """
    total_tags = 0.0
    correct_tags = 0.0
    correct_sentences = 0.0
    predicted_tags_list = []
    
    # For each sentence
    for i in range(len(data)):
        # Get the predicted tags
        predicted_tags = model.predict_tags_given_words(data[i])
        predicted_tags_list.append(predicted_tags)
        
        # For each predicted tag
        for index, pred_tag in enumerate(predicted_tags):
            # If both predicted and true tags are "O", ignore
            # If both tags are equal and not "O", count as a correct prediction
            if pred_tag == true_tags_dict[i][index] and pred_tag != corpus_tag_dict["O"]:
                correct_tags += 1
                # Count the tag
                total_tags += 1
            # When there has been a miss-prediction, sum up the total number of tags
            elif pred_tag != true_tags_dict[i][index]:
                total_tags += 1
        
        # Check if the whole sentence is exactly the same (including "O" tags)
        if list(predicted_tags) == true_tags_dict[i]:
            correct_sentences += 1
    
    return predicted_tags_list, correct_sentences, correct_tags, total_tags

def evaluate(data_, model, data_tag_pos, corpus_tag_dict, y_true, name, load=False):
    """
    Evaluate the model's performance on the given dataset and calculate various metrics.

    Parameters:
        data_ (list): A list of sentences.
        model (object): The trained model object.
        data_tag_pos (dict): A dictionary containing the true tags for each sentence.
        corpus_tag_dict (dict): A dictionary mapping tags to indices.
        y_true (list): A list of true tags.
        name (str): The name to use for saving results.
        load (bool): Whether to load previously saved predictions and metrics.

    Returns:
        None
    """
    if load == False:
        # Get predictions and accuracy metrics
        list_y_pred, correct_sentence, correct, total = get_accuracy(model, data_, data_tag_pos, corpus_tag_dict)
        y_pred = [tag for array in list_y_pred for tag in array]

        # Save predictions
        save_p("predictions/" + name+".pkl", y_pred)
    
        # Calculate F1 score and accuracy
        f1_score = f1_score_weighted(y_true, y_pred)
        acc = accuracy(y_true, y_pred)

        # Create a DataFrame to store metrics
        metrics = pd.DataFrame({
            "% of correct sentences": [correct_sentence / len(data_)],
            "% of tags correct": [correct / total],
            "F1 Score": [f1_score],
            "Accuracy": [acc]
        })
        
        # Save metrics to CSV file
        metrics.to_csv("results/" + name + ".csv", index=None)

    else:
        # Load previously saved metrics and predictions
        metrics = pd.read_csv("results/" + name + ".csv")
        y_pred = load_p("fitted_models/predictions/" + name+".pkl")
    
    # Display metrics
    print("Metrics:")
    display(HTML(round(metrics * 100, 2).to_html()))

    # Create a feasible dictionary for the confusion matrix
    feasible_dict = {k: corpus_tag_dict[k] for k in [{v: k for k, v in corpus_tag_dict.items()}[i] for i in set(y_true + y_pred)] if k in corpus_tag_dict}

    # Print confusion matrix
    print("\nConfusion Matrix\n")
    plot_confusion_matrix(y_true, y_pred, feasible_dict)
    
    print("\nMetrics by tag\n")
    display(HTML(pd.DataFrame(classification_report(
                y_true,y_pred,output_dict=True)).transpose().drop(["support"],axis=1).rename(
                index={str(v): k for k, v in corpus_tag_dict.items()}).to_html()))
    
    return(metrics)

############################################################################################################