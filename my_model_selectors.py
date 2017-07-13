import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.model_selection import KFold
        #print("SELECTOR CV STARTED 1")
        # TODO implement model selection using CV
        # split sequences in kfold 
        kfold_len = min(3, len(self.sequences))
        split_method = KFold(kfold_len)
        train_data_x = []
        train_data_lengths = []
        test_data_x = []
        test_data_lengths = []
        sequences = np.asarray(self.sequences)
        lengths = np.asarray(self.lengths)
        #print("sequences shape ", sequences)
        
        # number of state we might need to use
        n_best = self.min_n_components
        avg_best_score = 0
        # start the loop for experiment different model
        for comp_count in range(self.min_n_components, self.max_n_components + 1):
            # split the data in each fold 
            #print("Comp. Count: ",comp_count)
            avg_score = 0
            total_score = 0
            count = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    train_data_x =  np.vstack(sequences[cv_train_idx])
                    train_data_lengths = lengths[cv_train_idx]
                    test_data_x = np.vstack(sequences[cv_test_idx])
                    test_data_lengths = lengths[cv_test_idx]
                    #print ("Train data\n",train_data_x)
                    #print ("Train data squeeze\n",np.squeeze(train_data_x))
                    #print ("Train data lengths shape \n",train_data_lengths.shape)
                    #print ("train data lengths\n", train_data_lengths)
                    model = GaussianHMM(comp_count,
                                n_iter=1000,
                                random_state=self.random_state, 
                                verbose=False).fit(train_data_x,train_data_lengths)

                    #print ("test data\n",test_data_x)
                    #print ("Train data squeeze\n",np.squeeze(train_data_x))
                    #print ("test data lengths shape \n",test_data_lengths.shape)
                    #print ("test data lengths\n", test_data_lengths)
                    model_score = model.score(test_data_x, test_data_lengths)
                except:
                    continue # try next one 
                # if nothing is wrong then we accumulate the stats
                count += 1
                total_score += model_score
            # compare with best score
            if (count != 0): 
                avg_score = total_score / count
            # swap score if need to
            if (avg_best_score == 0) or (avg_score > avg_best_score):
                avg_best_score = avg_score
                n_best = comp_count
        # now return the best model 
        return self.base_model(n_best)

        
                

           
        
        
        
        
        
        
        
        
        
        
        
