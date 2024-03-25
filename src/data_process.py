# -*- coding: utf-8 -*-
# Copyright 2024 Jingjing Zhai, Minggui Song.
# All rights reserved.
#

"""Represent a collect flnc information.

What's here:

Extract sample sequences features.
-------------------------------------------

Classes:
    - DataProcess
"""

from distutils.command.config import config
from logging import getLogger
from src.sys_output import Output
import numpy as np
from pathlib import Path
from Bio import SeqIO
import numpy as np
import h5py

np.random.seed(1)
logger = getLogger(__name__)  # pylint: disable=invalid-name


class DataProcess(object):
    """The preprocess positive and negative data sets process.

    Attributes:
        - args: Arguments.
        - output: Output info, warning and error.

    """
    def __init__(self, arguments) -> None:
        """Initialize CollectFlncInfo."""
        self.args = arguments
        self.output = Output()
        self.output.info(
            f'Initializing {self.__class__.__name__}: (args: {arguments}.')
        logger.debug(
            f'Initializing {self.__class__.__name__}: (args: {arguments}.')
    
    def writeh5(self,seqs,filename,inputwindow,offset_values=None):
        seqsnp=np.zeros((len(seqs),4,inputwindow),np.bool_)

        mydict={'A':np.asarray([1,0,0,0]),'C':np.asarray([0,1,0,0]),'G':np.asarray([0,0,1,0]),'T':np.asarray([0,0,0,1]),'N':np.asarray([0,0,0,0]),'H':np.asarray([0,0,0,0]),'M':np.asarray([0,0,0,0]),'Y':np.asarray([0,0,0,0]),'R':np.asarray([0,0,0,0]),'K':np.asarray([0,0,0,0]),'W':np.asarray([0,0,0,0]),'S':np.asarray([0,0,0,0]),'D':np.asarray([0,0,0,0]),'E':np.asarray([0,0,0,0]),'a':np.asarray([1,0,0,0]),'c':np.asarray([0,1,0,0]),'g':np.asarray([0,0,1,0]),'t':np.asarray([0,0,0,1]),'m':np.asarray([0,0,0,0]),'n':np.asarray([0,0,0,0]), 'e':np.asarray([0,0,0,0])}
        n=0
        offset_values = np.zeros(len(seqs))
        for line,o in zip(seqs,offset_values):
            if len(line)<inputwindow:
                continue
                raise Exception("Each fasta sequence has to be at least 1000bp.")
            #if the user specified region/sequence is longer than 1000bp, use the center 1000bp
            cline = line[int((int(o) + (len(line)-inputwindow)/2)):int(int(o)+(len(line)+inputwindow)/2)]
            for c,i in zip(cline,range(len(cline))):
                seqsnp[n,:,i]=mydict[c]
            n=n+1
        seqsnp=seqsnp[:n,:,:]
        seqsnp = seqsnp.astype(np.uint8)
        np.savez_compressed(filename, a=seqsnp)

    def process(self) -> None:
        """Call the preprocessing data process object."""
        self.output.info('Starting preprocessing data Process.')
        logger.debug('Starting preprocessing data Process.')

        inputwindow=int(self.args.window_length)
        fasta_sequences = SeqIO.parse(open(self.args.input),'fasta')
        seqs=[str(fasta.seq) for fasta in fasta_sequences]
        self.writeh5(seqs,self.args.output,inputwindow)
        
        self.output.info('Completed preprocessing data Process.')
        logger.debug('Completed preprocessing data Process.')
