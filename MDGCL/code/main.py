import torch
import numpy as np
from parms_setting import settings
from data_preprocess import load_data
from train import train_model

# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print('args.cuda',args.cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

aurocs, prcs, f1s= [],[],[]
data_s, data_f, train_loader, test_loader = load_data(args, n_splits=5)

for fold, (train_loader, test_loader) in enumerate(zip(train_loader, test_loader)):
    print(f"Training on fold {fold+1}")
    auroc, prc, f1, loss = train_model(data_s, data_f, train_loader, test_loader, args, fold)
    aurocs.append(auroc)
    prcs.append(prc)
    f1s.append(f1)


print('aurocs:',aurocs)
print('prcs:',prcs)
print('f1s:',f1s)
print('np.mean(aurocs):', np.mean(aurocs),'np.std(aurocs):',np.std(aurocs))
print('np.mean(prcs):', np.mean(prcs),'np.std(prcs):',np.std(prcs))
print('np.mean(f1s):', np.mean(f1s),'np.std(prcs):',np.std(f1s))



