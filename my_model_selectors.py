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
        best_bic = None
        best_model = None
        # loop through all model and calculate BIC
        for comp_count in range(self.min_n_components, self.max_n_components + 1): 
            # BIC = -2 * logL + free_parameters * log(number of model states)
            BIC = None
            model = None
            try:    
                # get the base model for the specific topology
                # base model already call fit 
                model = self.base_model(comp_count)
                #print ("Finish fit model")
                # now we need to calculate BIC
                logL = model.score(self.X, self.lengths)
                #print ("LogL: ",logL)
                # free_parameters = transistion probs(n*n) + means(n*f) + covars(n*f) - 1
                # f: number of features used to train the model
                free_parameters = comp_count**2 + 2 * comp_count * model.n_features - 1
                #print ("Free parameters: ",free_parameters)
                BIC = -2 * logL + free_parameters * math.log(comp_count)
                #print ("BIC ",BIC)
            except:
                # ignore bad model
                #print("Bad model. Try next one. n: ", comp_count)
                continue
            # update best model as the smaller BIC the better 
            if (best_bic == None) or (BIC < best_bic):
                #print("Update model for word: ", self.this_word)
                best_bic = BIC
                best_model = model

        return best_model


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
        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        # DIC = logL(this_word) - 1 / (total_words - 1) SUM: all logL(other_words but this_word)
        best_dic = None
        best_model = None
        total_words = len(self.words.keys())
        for comp_count in range(self.min_n_components, self.max_n_components + 1): 
            
            model = None
            DIC = None
            try:
            # fit the model for self.X and self.lengths
                model = self.base_model(comp_count)
                logL_this_word = model.score(self.X, self.lengths)
            # now calculate the sum of all other words liklihood
                others_logL_sum = 0
                for word in self.words.keys():
                    # skip this_word
                    if word == self.this_word:
                        continue
                    # get likelihood and sum it
                    other_X, other_lengths = self.hwords[word]
                    others_logL_sum += model.score(other_X, other_lengths)
            # not checking division by zero here 
            # as it will hit the except case and we will ignore 
            # the model .. this should not happen probably
                DIC = logL_this_word - 1 / (total_words - 1) * others_logL_sum
            except:
                continue

            # higher DIC score means better discriminant model which
            # improved accuracy for classification 
            if (best_dic == None) or (DIC > best_dic):
                best_dic = DIC
                best_model = model

        # return the best model 
        return best_model


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
        avg_best_score = None
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
                    model = GaussianHMM(comp_count,
                                n_iter=1000,
                                random_state=self.random_state, 
                                verbose=False).fit(train_data_x,train_data_lengths)

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
            if (avg_best_score == None) or (avg_score > avg_best_score):
                avg_best_score = avg_score
                n_best = comp_count
        # now return the best model 
        return self.base_model(n_best)

        
                

           
        
        
        
        
        
        
        
        
        
        
        
