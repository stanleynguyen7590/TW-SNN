#####################################
#   @author: Nitin Rathi    		#
#	Writes commands to execute		#
# 	in script.sh				 	#
#####################################

import argparse
import itertools
import os
import pdb

from scipy.special import comb

hyperparameters = {
    'architecture'	:	{'VGG5','VGG9','VGG11','VGG13','VGG19','RESNET12','RESNET20','RESNET34'},
    'learning_rate'	:	{'1e-2'},
    'epochs'		:	{'100'},
    'lr_interval'	:	{'\'0.60 0.80 0.90\''},
    'lr_reduce'		: 	{'10'},
    'dataset'		:	{'CIFAR10'},
    'batch_size'	:	{'64'},
    'optimizer' 	: 	{'SGD'},
		'weight_decay':{'1e-4'},
    'dropout'		:	{'0.3'}
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='create script for running a file')
    parser.add_argument('--filename', 		default='ann.py',
                        help='python filename to run')
    parser.add_argument('--parallel',		action='store_true',
                        help='whether to allow all combinations to run simultaneously')
    args = parser.parse_args()

    f = open('script.sh', 'w', buffering=1)
    f.write('#!/bin/bash')
    f.write('\n')

    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print('Total possible combinations: ', len(combinations))
    for c in combinations:
        s = ''
        for key, value in c.items():
            s = s+'--'+key+' '+value+' '

        s = 'python '+args.filename+' '+s
        s = s+'--log '
        if args.parallel:
            s = s + '& '
        f.write('\n')
        f.write(s)

    f.close()
    # os.system('./script.sh')
