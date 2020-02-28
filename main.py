import sys
import yaml
import pprint

from train import train as train
from train import train_mmd as train_mmd


########################################################################
# configuration loading
########################################################################

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


configuration = get_config(sys.argv[1])
action = configuration['action']
print('the configuration used is:')
pprint.pprint(configuration, indent=4)

########################################################################
# main method
########################################################################

if __name__ == '__main__':
    if action == 'train':
        print('starting main training loop with train configuration')
        train(configuration)

    if action == 'train_mmd':
        print('starting main training loop with MMD-loss configuration')
        train_mmd(configuration)
