#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2016 DARPA.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

import numpy as np

class ProbabilisticStateMachine(object):
    '''
    Class for probabilistic state machine
    '''
    def __init__(self, N, initial_state=None, seed=None):

        # store off inputs to be used during start()
        self._initial_state = initial_state
        self._seed = seed
        self._N = N
        self._stick_to_path_prob = 0.6

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

        return self.state


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

        return rand_ind
