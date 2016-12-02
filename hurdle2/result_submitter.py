# this module will be imported in the into your flowgraph

import random

from thrift import Thrift
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

from hurdle2_rpc import Hurdle2Scoring
from hurdle2_rpc.ttypes import BinContents as BC

# REQUIRED IMPORTS 
from UNT_hurdle2 import UNT_hurdle2
from sklearn.externals import joblib
import pandas as pd
import scipy
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def submit_my_answer(answer, host, port):

    success = False
    try:

        # Make socket
        transport = TSocket.TSocket(host, port)

        # Buffering is critical. Raw sockets are very slow
        transport = TTransport.TBufferedTransport(transport)

        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(transport)

        # Create a client to use the protocol encoder
        # client = Calculator.Client(protocol)
        client = Hurdle2Scoring.Client(protocol)

        # Connect!
        transport.open()

        # send answer to the scoring server
        success = client.submitAnswer(answer)

        # Close!
        transport.close()

    except Thrift.TException as tx:
        print('%s' % tx.message)

    return success

def make_random_guess(num_bins):

    choices = [BC.NOISE, BC.FM, BC.GMSK, BC.QPSK]
    answer = {}

    for i in range(num_bins):
        answer[i] = random.choice(choices)

    return answer

def make_prediction(sample_file):
    h2 = UNT_hurdle2('clf/RF.pkl','clf/LabelEncoder.pkl')
    answer = h2.make_prediction(sample_file).tolist()

    return answer


