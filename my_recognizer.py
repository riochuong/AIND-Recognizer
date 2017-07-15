import warnings
from asl_data import SinglesData
from math import inf as INF


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    #print (test_set.wordlist)

    # iterate through word to find the best word
    for word_id, model_word in enumerate(test_set.wordlist):
      # if we cannot find a model for the key
      prob_dict = {}
      best_word_id = None
      best_score = None
      best_word = None
      x, lengths = test_set.get_all_Xlengths()[word_id]
      
      # spin through all the model to check which word is 
      # this best match
      for key_word in models.keys():
        score = None
        try:
          model = models[key_word]
          score = model.score(x,lengths)
        except:
          # assign super small value for not matching anything word
          score = -INF
        # update prob dict
        prob_dict[key_word] = score
        # update best word
        if (best_score == None) or (score > best_score):
          best_score = score
          best_word_id = word_id
          best_word = key_word

      # now update the guesses dict
      if (best_score != None):
        probabilities.append(prob_dict)
        guesses.append(best_word)


    return (probabilities, guesses)