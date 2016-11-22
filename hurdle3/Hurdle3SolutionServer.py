#!/usr/bin/python
# encoding: utf-8

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import os
import sys

from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport

from hurdle3.RandomGuesser import RandomGuesser
from hurdle3_rpc import Hurdle3Execution
from hurdle3_rpc.ttypes import StepResult
from UNT_hurdle import JorgePlayer


class SolutionHandler:

    def __init__(self, num_states=10, seed=None):
        self.log = {}

        # Change RandomGuesser to your solution
        self.solution = JorgePlayer(num_states, retrain=10000)

    def start(self):

        self.solution.restart()
        prediction, next_state = self.solution.start()

        return StepResult(prediction, next_state)

    def step(self, reward, observation):

        prediction, next_state = self.solution.step(reward, observation)

        return StepResult(prediction, next_state)

    def stop(self):
        sys.exit()

def main(argv=None):  

    try:
        # Setup argument parser
        parser = ArgumentParser(description="Hurdle 3 Example Solution", formatter_class=ArgumentDefaultsHelpFormatter)

        parser.add_argument("--host",          type=str, default="0.0.0.0",      help="IP address that the solution server will listen for connections on")  
        parser.add_argument("--rpc-port",      type=int, default=9090,           help="Port for RPC connections") 
        parser.add_argument("--seed",          type=int, default=None,           help="Random number generator seed to use for repeatable tests")

        # Process arguments
        args = parser.parse_args()


        handler = SolutionHandler(seed=args.seed)
        processor = Hurdle3Execution.Processor(handler)
        transport = TSocket.TServerSocket(host=args.host, port=args.rpc_port)
        tfactory = TTransport.TBufferedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

        print('Starting the server...')
        server.serve()


    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except SystemExit:
        print('done.')
        return 0

if __name__ == "__main__":

    sys.exit(main())
