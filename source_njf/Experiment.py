import warnings
from abc import ABC, abstractmethod

import args_from_cli
from train_loop import load_network_from_checkpoint, MyNet
import torch
from UVEncoder import UVEncoder

import os
import json
from train_loop import main

class Experiment(ABC):
    '''
    base class for experiments
    '''

    def __init__(self,name,description, cpuonly=False):
        self.net = None
        self.name = name
        self.description = description
        self.cpuonly = cpuonly

    def modify_args(self,args):
        '''
        called before setting args, to enable modifying args
        :param args: the original args
        :return: the args, with any modification that should be included
        '''
        return args

    def get_encoder(self,args):
        '''
        initialize the encoder for this experiment and return it
        :param args: the cli args
        :return: the encoder, initialize
        '''
        args = self.modify_args(args)
        encoder = UVEncoder(args)
        self.init_encoder(encoder, args)
        return encoder

    @abstractmethod
    def init_encoder(self,encoder,args):
        '''
        abstract method that should be overridden to init the encoder object
        :return: DeformationEncoder object
        '''
        pass

    def load_network(self,checkpoint_path):
        '''
        load a network from a checkpoint, and store it internally
        :param checkpoint_path: path to checkpoint
        :return: the network loaded from the checkpoint
        '''
        self.net = load_network_from_checkpoint(checkpoint_path, cpuonly=self.cpuonly)
        if not self.cpuonly:
           self.net.cuda(0)
        else:
            self.net.cpu()
        self.__args = self.net.args
        return self.net

    def get_args_and_train(self, args):
        if self.net is not None:
            warnings.warn("seems like you loaded a network, but are now running training -- FYI, the loaded network is not being used in training (you need to specify the checkpoint in CLI")

        args = self.modify_args(args)
        self.args = args

        # Change name based on the cli arg
        self.name = self.args.expname
        print(f"starting training with args: {args}")

        if not args.continuetrain and not args.test:
            gen = self.get_encoder(args)
        else: # Load latest checkpoint model based on checkpoints folder in output path
            import re
            checkpointdir = os.path.join(self.args.outputdir, self.args.expname, "ckpt")
            if os.path.exists(checkpointdir):
                maxepoch = 0
                maxstep = 0
                checkpoint = None
                for file in os.listdir(checkpointdir):
                    if file.endswith(".ckpt"):
                        result = re.search(r"epoch=(\d+)-step=(\d+)", file)
                        epoch = int(result.group(1))
                        step = int(result.group(2))

                        if epoch > maxepoch:
                            maxepoch = epoch
                            maxstep = step
                            checkpoint = os.path.join(checkpointdir, file)
                        elif epoch == maxepoch and step > maxstep:
                            maxstep = step
                            checkpoint = os.path.join(checkpointdir, file)

                if checkpoint is not None and os.path.exists(checkpoint):
                    print(f'************************** STARTING TRAINING FROM CHECKPOINT {checkpoint}' )
                    gen = checkpoint
                else:
                    print(f"No checkpoint found at {checkpointdir}!")
                    gen = self.get_encoder(args)
            else:
                print(f"No checkpoint found at {checkpointdir}!")
                gen = self.get_encoder(args)

        if args.test:
            name = os.path.join(self.name,'test')

        main(gen, args)