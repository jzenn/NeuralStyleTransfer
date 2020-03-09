import os
import sys
import yaml
import pprint

from train import train as train
from train import train_mmd as train_mmd

# path to python_utils
sys.path.insert(0, '../utils')
sys.path.insert(0, '/home/zenn')

from python_utils.FolderStructure import FolderStructure
from python_utils.UniqueName import UniqueName


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

# generate folder structure
configuration['folder_structure'] = FolderStructure(name='nst', working_directory='./experiments')
configuration['folder_structure'].create_structure(['images', 'loss'])

configuration['image_saving_path'] = os.path.join(configuration['folder_structure'].get_parent_folder(), '/images')
configuration['unique_name'] = UniqueName(name='nst_image', configuration=configuration).get_unique_identifier()

configuration['content_layers'] = [configuration['style_layers'][-1]]



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
