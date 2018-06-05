
# coding: utf-8

# In[34]:


import sys, numpy as np
#from global_vars import *

VERBOSE = True
DATA_NORMALIZE = False

STONE_BLANK = 0.0
STONE_HUMAN = 1.0
STONE_AI = -1.0

FIELD_WIDTH = 7
FIELD_HEIGHT = 6
CONNECT = 4

WIN = 0.0
DRAW = 0.5
LOSS = 1.0

DATA_NUM_ATTR = 43

def my_converter( x ):
    """ Converter for Numpys loadtxt function.
    Replace the strings with float values.
    """

    if x == 'x': return STONE_HUMAN
    elif x == 'o': return STONE_AI
    elif x == 'b': return STONE_BLANK
    elif x == "win": return WIN
    elif x == "loss": return LOSS
    elif x == "draw": return DRAW


def normalize( data ):
    """ Normalize given data. """

    sys.stdout.write( " Normalizing data ..." )
    sys.stdout.flush()

    data -= data.mean( axis = 0 )
    imax = np.concatenate(
        (
            data.max( axis = 0 ) * np.ones( ( 1, len( data[0] ) ) ),
            data.min( axis = 0 ) * np.ones( ( 1, len( data[0] ) ) )
        ), axis = 0 ).max( axis = 0 )
    data /= imax

    return data


def import_traindata( file_in ):
    """ Import the file with training data for the AI. """

    sys.stdout.write( "Importing training data ..." )
    sys.stdout.flush()

    # Assign converters for each attribute.
    # A dict where the key is the index for each attribute.
    # The value for each key is the same function to replace the string with a float.
    convs = dict( zip( range( DATA_NUM_ATTR + 1 ) , [my_converter] * DATA_NUM_ATTR ) )
    connectfour = np.loadtxt( file_in, delimiter = ',', converters = convs )

    cf_original = []
    f = open( file_in, "r" )
    for line in f:
        row = line.split( ',' )
        row[len( row ) - 1] = row[len( row ) - 1].replace( '\n', '' )
        cf_original.append( row )

    # Split in data and targets
    data = connectfour[:,:DATA_NUM_ATTR - 1]
    targets = connectfour[:,DATA_NUM_ATTR - 1:DATA_NUM_ATTR]

    if DATA_NORMALIZE:
        data = normalize( data )

    sys.stdout.write( " Done.\n\n" )

    return data, targets, cf_original


# # Train MLP (Neural Network) with UCI data

# In[35]:


from sklearn.neural_network import MLPClassifier, MLPRegressor

data, targets, original_import = import_traindata("connect-4.data")
training_data = data[:60000,:]
training_targets = targets[:60000,:]
np.ravel(targets)

clf = MLPRegressor(solver='adam')
clf.fit(training_data, training_targets)


# # Code to Test Validation Error and Create Data Samples

# In[36]:


from sklearn.neural_network import MLPClassifier, MLPRegressor

data, targets, original_import = import_traindata("connect-4.data")
training_data = data[:60000,:]
training_targets = targets[:60000,:]
np.ravel(targets)

clf = MLPRegressor(solver='adam')
clf.fit(training_data, training_targets)


# In[121]:


error = 0
for x in range(60000,data.shape[0]):
    a = data[x]
    #a.reshape(1,-1)
    pred = clf.predict([a])[0]
    error = error + np.square(targets[x] - pred)
    #print("TARGET IS " + str(targets[x]) + "PREDICT IS " + str(pred))
error/7557

