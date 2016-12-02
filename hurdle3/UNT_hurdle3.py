# coding: utf-8

# ## Darpa Player 
# 
# The darpa player is defined by a Markov Chain Model with N states and the output can be int in the range of 1:M 
# 
# My Class attemps to recreate the DARPA Player so we can use it to simulate and run MonteCarlo simulations against our model and see if we can Pass the Hurdle #3 on the DARPA competition 

# In[1]:

import random
import numpy as np
import pandas as pd


# In[2]:

class RandomPlayer:
    ''' A Simple class that simulate a Random Player for the #3 Hurdle of the DARPA Competition  '''
    import numpy as np
    
    def normalizeProbs(self,p_):        
        
        if (p_<0).any():
            print "Error", p_
        p_ = [float(i)/sum(p_) for i in p_]            
        return p_
        
    def createProbs(self,L):
        p_ = np.random.random(self.N)
        
        if (p_<0).any():
            print "Error 5", p_
            e
        p_ = self.normalizeProbs(p_)                
        
        return p_
        
    def __init__ (self, Max_N, M):
        self._type = "Random"
        self.N = np.random.choice(range(3,Max_N)) ## Random N-states
        self.M = M                               ##  M-outputs
        self.q = np.random.choice(range(self.N)) ## Random current state given N States 
        self.Q = np.zeros(shape= (self.M-1, self.N)) ## Transition Matrix Given Other Player Output
        
        # Iterate all rows 
        for i in range(self.Q.shape[0]-1):
            
            p_ = self.createProbs(self.N)
            self.Q[i,:] = p_ # create row probabilities 
            # Get Row Q[q,:] row q all colums
            # Get Col Q[:,q] all rows q colum
            
        
        # In order to get A -> choice from Q column
    def start(self):
        
        p_ = self.Q[:,self.q]
        p_ = self.normalizeProbs(p_)        
            
        return [np.random.choice(range(1,self.M), p=p_),self.predict()]
        
    def step(self, A):
        
        ## TODO ERROR CASE WHEN A > M 
        
        # Return A
        
        p_ = self.Q[:,self.q]
                  
        p_ = self.normalizeProbs(p_)
        
        D =  np.random.choice(range(1,self.M), p=p_)
        
        # update the state N 
        self.update(D)
        
        return [D,self.predict()]
        
    def update(self, A):
        #print self.Q.shape, A-1
        
        p_ = self.Q[A-1,:]               
            
        p_ = self.normalizeProbs(p_)
        #p_ = p_ / sum(p_)
        
        self.q = np.random.choice(range(self.N), p=p_)
        
        
    def predict(self):
        return np.random.choice(range(self.M)) # Random Guess
        
    def restart(self, Max_N):
        self.__init__(Max_N,self.M)
        return True
        
        
         
       
class DarpaPlayer(object):
    '''
    Modified Class for probabilistic state machine 
    Changes are in order to make it play in our simulation
    '''
    def __init__(self, N, initial_state=None,probs = 0.6 ,seed=None):

        # store off inputs to be used during start()
        self._initial_state = initial_state
        self._seed = seed
        self._N = N
        self._stick_to_path_prob = probs
        self.start()

    def _generate_transition_matrices(self, num_states, stick_to_path_prob):
        '''
        For a given number of state machine states
        and seeded random number generator, make
        the transition matrices
        '''

        path_length = int(np.floor(num_states / 2))

        # build up a set of random paths based on the
        # current state machine state and the state machine
        # input
        random_paths = np.zeros((num_states, path_length))

        for k in range(num_states):
            # get a random path through the state space
            # for each state machine input
            # current_path = range(N)
            # rng.shuffle(current_path)

            current_path = self._rng.permutation(num_states)
            # truncate path to path_length and store
            random_paths[k, :] = current_path[:path_length]

        for i in range(num_states):  # loop over state machine states
            for k in range(num_states):  # loop over state machine inputs

                # is the current state machine state part of the path
                # for this state machine input?
                this_ind_in_seq = np.argwhere(random_paths[k, :] == i)

                # if the state is in the current path, get the
                # path's next state
                if this_ind_in_seq.size > 0:
                    # extract the scalar index
                    this_ind_in_seq = this_ind_in_seq[0, 0]

                    # if we're at the end of the path, go back to start
                    if this_ind_in_seq == path_length - 1:
                        next_state = int(random_paths[k, 0])
                    # otherwise go to next state in the path
                    else:
                        next_state = int(random_paths[k, this_ind_in_seq + 1])
                # otherwise signal that the state wasn't in the path
                else:
                    next_state = None

                # compute the transition matrix for the current state
                # and state machine input
                if next_state is not None:

                    # set up transition probabilities such that
                    # there is a stick_to_path_prob chance of going
                    # to the next state in the path and a
                    # (1-stick_to_path_prob) of doing anything else
                    transition_probs = self._rng.rand(num_states)
                    transition_probs[next_state] = 0

                    # normalize probabilities
                    transition_probs = transition_probs / sum(transition_probs)
                    transition_probs = (1 - stick_to_path_prob) * transition_probs

                    # set the stick_to_path_prob
                    transition_probs[next_state] = stick_to_path_prob

                # if the current state wasn't in the path, make the
                # next state transition purely random
                else:
                    # normalize the transition probabilities
                    transition_probs = self._rng.rand(num_states)
                    transition_probs = transition_probs / sum(transition_probs)

                self.transition_mat[i, k, :] = transition_probs

                # precomputing for better speed
                self.cum_sum_mat[i, k, :] = np.cumsum(transition_probs)

    def start(self):
        '''
        Run first iteration as a special case
        '''

        # set up a dedicated random number generator
        # for this object to guarantee repeatability
        # if a seed is specified
        self._rng = np.random.RandomState(self._seed)

        # dimensions are current state by state machine input by
        # possible next state
        self.transition_mat = np.zeros((self._N, self._N, self._N))

        # dimensions are current state by state machine input by
        # possible next state. Precomputing cumulative sums to
        # speed up execution
        self.cum_sum_mat = np.zeros((self._N, self._N, self._N))

        # set up transition matrices
        self._generate_transition_matrices(self._N, self._stick_to_path_prob)


        if self._initial_state is not None:
            self.state = self._initial_state
        else:
            self.state = int(self._rng.randint(self._N))

        return self.state, np.random.choice(range(1,self._N))


    def step(self, observation):
        '''
        Given the observation, generate the next probabilistic action
        '''

        # get the transition probabilities given the current state and
        # state machine input
        # transition_probs = self.transition_mat[self.state, observation, :]

        # Note, we could probably speed up execution by precomputing all cumulative sums,
        # but I'll get back to that if it becomes an issue
        # cum_sums = np.cumsum(transition_probs)

        # trying out precomputed cumulative sums for speed
        cum_sums = self.cum_sum_mat[self.state, observation, :]

        # print("cumulative sums {}".format(cum_sums))

        rand_val = self._rng.rand()

        rand_ind = np.argwhere(cum_sums > rand_val)
        # print("rand_ind {}".format(rand_ind))

        # handle edge conditions
        if rand_ind.size == 0:
            rand_ind = len(cum_sums) - 1

        # otherwise extract the relevant index
        else:

            rand_ind = rand_ind[0, 0]

        self.state = rand_ind

        return rand_ind, np.random.choice(range(1,self._N))

    def restart(self, Max_N):
        self.__init__(Max_N,probs=self._stick_to_path_prob)
        return True


class JorgePlayer:
    ''' 
    A Simple class that simulate a base Player based on Jorge's prediction
    for the #3 Hurdle of the DARPA Competition
    '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier 
    import itertools
   
    def __init__ (self, N,retrain=500):
        
        self._type = "Jorge"
        self.N = N                        ## Random N-states (NOT USED)
        self.M = N                        ##  M-outputs        
        self.retrain = retrain            ## Retrain after #retrain turns of the game        
        self.Dataset, self.BestPlayTable = self.createDataset()
        self.record = pd.DataFrame( columns=['Player_Last','Last','pred','Player_Play'])
        
        self.turn = 0
        self.PlayerLast = -1
                                    
    def start(self):

        D,P = [np.random.choice(range(self.M)),np.random.choice(range(self.M))]
        self.Prev_D = D
        return D,P
        
    def step(self,reward ,Last):
        
        self.turn = self.turn +1
        
        if self.turn < self.retrain:
            PlayerPlay,pred = self.start()
                    
        else:
            # check if is time to train 
            if self.turn%self.retrain == 0:
                # Train
                self.Train()
                    
            PlayerPlay,pred = self.predict(Last)            
                 
        self.appendData(Last,self.PlayerLast, pred,PlayerPlay)

        self.PlayerLast = PlayerPlay
        
        return [pred,PlayerPlay]
    
    
    def updateBestPlay(self,Player_Last, Last):
        if PlayerPlay == Last:
            self.BestPlayTable[self.BestPlayTable.index == Player_Last] = self.BestPlayTable[self.BestPlayTable.index == Player_Last] -1
        else:
            self.BestPlayTable[self.BestPlayTable.index == Player_Last] = self.BestPlayTable[self.BestPlayTable.index == Player_Last] +1
            
    
    def updateDataset(self):  
        
        df= self.record
        df['Label'] = df.Last.shift(-1)
        df['prob'] = df.Last.shift(-1)
        self.Dataset =df.groupby(['Player_Play','Last']).Label.agg(lambda x:x.value_counts().index[0]).reset_index()                                
        dataset2 = df.groupby(['Player_Play','Last']).prob.agg(lambda x:np.float(x.value_counts().values[0])/x.value_counts().values.sum()).reset_index()
        self.Dataset = self.Dataset.merge(dataset2, on=['Player_Play','Last'])  
        
    def createDataset(self):                
        
        df = pd.DataFrame(columns=['Player_Play','Last','pred','Player_Last','pred'])
        #table to keep score of our best plays 
        table = pd.DataFrame(index=range(self.N), columns=['Points'])
        return df, table
    
    
    def appendData(self, Last, Player_Last,pred,Play):
                
        self.record = self.record.append(pd.DataFrame([[Last, Player_Last,pred,Play]], columns=['Last','Player_Last','pred','Player_Play']),ignore_index=True)
        
    def Train(self):                
        
        print "Training.. "
        # Training is really update the Dataset
        self.updateDataset()
        
        return True    

    
    def predict(self,Last):
        
        good = False
        for i in range(1,11):
            play = self.Dataset[(self.Dataset.Last == Last) & (self.Dataset.Player_Play != self.Dataset.Label)]\
            .sort(columns= 'prob', ascending = False)\
            .head(1).Player_Play.iloc[0]
            
            pred = self.Dataset[(self.Dataset.Player_Play == play) & (self.Dataset.Last == Last)].Label.iloc[0]
            
            if play != pred:
                good = True
                break
        
        if not good:
            play,pred = np.random.randint(0,self.N,size=2)               

        
        return play,pred # P = More Likely, D = Less Likely 
        
    def restart(self):
        print "Restarting.. "
        self.__init__(self.N,self.retrain)
        return True
                           
                               
class Hurdle_MC:
    ''' Simple Class to Emulate the 3rd hurdle of the DARPA Competition'''
    
    
    
    def normalizeProbs(self,p_):
        
        #p_ = p_ / sum(p_)   
        p_ = [float(i)/sum(p_) for i in p_]
        return p_
        
    
    def __init__ (self, Player_1, Player_2, total_turns=1000,total_runs=10, Max_N=10, Max_M=10):
        import numpy as np
        import pandas as pd 
        
        self.turns = total_turns
        self.MaxN = Max_N  ## Variable for Random Player
        self.MaxM = Max_M ## Variable for Random Player
        self.PlayerD = Player_1 # 'DARPA'
        self.PlayerA = Player_2 # 'Us'
        self.BestPlayer = Player_2 ## Save the best player (for intelligent players)
        self.mc_runs = total_runs # Total MonteCarlo runs
        self.score = np.zeros(self.turns) # current score
        self.LastScore = np.ndarray(shape= (total_runs,self.turns)) # Save the score of each run 
        self.BestPlayer = None
        
        
    def _Score(self,Play,pred,darpa_state):
        payoff = 0 
        if Play==darpa_state:  
            payoff = payoff-12
        if Play!=darpa_state:
            payoff = payoff+1
        if pred==darpa_state:
            payoff = payoff+3
            
        return payoff
    
    def one_game(self):
        
        
        self.PlayerD.restart(self.MaxN)
        if self.PlayerA._type == "Random":
            self.PlayerA.restart(self.MaxN)
        else:            
            self.PlayerA.restart()
        
        [P,A] = self.PlayerA.start()
        
        for turn in range(self.turns-1):# All turns but the last turn
            [D,x] = self.PlayerD.step(A) # DARPA Turn with given output A
            self.score[turn] = self._Score(A,P,D)# Score previous turn
            [P,A] = self.PlayerA.step(self.score[turn-1],D) # Make the play for next turn
            
        self.score[self.turns-1] = self._Score(A,P,D)
        
        
        return self.PlayerA
        
    
    def plot_games(self):
        
        import matplotlib.pyplot as plt
        import pandas as pd
        import math 
        get_ipython().magic(u'matplotlib inline')
        
        # Convert the Scores of each run to DataFrame and get 10 means for ploting
        score_df = pd.DataFrame(self.LastScore).transpose()#[-math.ceil(self.turns)/30:]
        last = np.int32(math.ceil(self.turns/30)) 
        
        score_df = score_df[-last:].cumsum()
        score_df = self.bin_runs(score_df)
        
        #plt.plot(score_df, label=score_df.columns)
        
        
        score_df.plot()        
        l= plt.axhline(y=last*2,xmin=0, xmax=1 ,linewidth=2, color='ForestGreen',label='Threshold')
        plt.ylabel('Score')
        plt.xlabel('Turn')        
        #plt.title("Random Player Score over Time (Turns) for M = {0}".format(self.PlayerD.M))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
    
    def bin_runs(self, df):
        import pandas as pd
        df_avg = pd.DataFrame()
        bins = np.int16(np.round(np.linspace(0,len(df.columns),num=11)))
        
        for i in range(len(bins)-1):
            df_avg[i] = df[df.columns[i:i+1]].mean(axis=1)

        return df_avg
    
    def UpdateBestPlayer(self, Player):
        
        import pandas as pd
        import math 
        
        last = np.int32(math.ceil(self.turns/30)) 
        current_score = self.score[-last:].cumsum()[-1]
        
        if self.BestPlayer == None:
            self.BestPlayer = Player
            self.BestScore = current_score

        else:                        
            if current_score > self.BestScore:
                self.BestPlayer = Player
                self.BestScore = current_score
            
        return True
        
    def simulate(self):
                    
        for ith_game in range(self.mc_runs):
            #print ith_game
            #M_ = np.random.choice(range(3,self.MaxM)) ## Random M-channels
            ith_Player = self.one_game()            
            self.LastScore[ith_game] = self.score
            self.UpdateBestPlayer(ith_Player)
            
        self.plot_games()    
     
     
        
        
