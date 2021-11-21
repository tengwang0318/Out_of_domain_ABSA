"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --models-path=<str>                path of the trained models
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --bert-type=<str>                      language choice [default: base-bert]
    --real-test-path=<str>                 file path of the real test set [default: ]
    --fake-test-path=<str>              file path of the fake test set [default: ]
"""
from learner import AllAspectsTest
from model import OodModel
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np

args = docopt(__doc__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Define Dataloaders
#####################################################################
real_test_dataset = DataClass(args, args['--real-test-path'])
fake_test_dataset = DataClass(args, args['--fake-test-path'])
real_test_data_loader = DataLoader(real_test_dataset,
                                   batch_size=int(args['--test-batch-size']),
                                   shuffle=False)
fake_test_data_loader = DataLoader(fake_test_dataset,
                                   batch_size=int(args['--test-batch-size']),
                                   shuffle=False)

print('The number of Test batches: ', len(fake_test_data_loader))
#############################################################################
# Run the models on a Test set
#############################################################################
model = OodModel(model_type=args['--bert-type'])
learn = AllAspectsTest(model, fake_test_data_loader,real_test_data_loader, model_path='models/' + args['--models-path'])
learn.predict(device=device)
