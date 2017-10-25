import numpy as np
X={}
for i in range(0,5000):
    X[str(i)]=np.random.rand(100)

import pickle
datafile = r'D:\JanneK\Documents\git_repos\text_classification\Project\results\test.pickle'
with open(datafile, "wb") as f:
    pickle.dump(X,f, pickle.HIGHEST_PROTOCOL)

datafile = r'D:\JanneK\Documents\git_repos\text_classification\Project\results\test.pickle'
with open(datafile, "rb") as f:
    pickle.load(f)

# import shelve
# filename=r'D:\JanneK\Documents\git_repos\text_classification\Project\results\test.dat'
# my_shelf = shelve.open(filename,'n') # 'n' for new



print('test')
