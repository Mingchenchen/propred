import numpy as np
import pandas as pd 
import os, glob, bcolz 
from parseData import * 
import multiprocessing as mp 

def threeGramEncodeAA(df, code, filename):
    ## TODO: Account for end of a protein sequence (and multimeric proteins) 
    total = []
    for j in [0,1,2]:
        X = []
        for i in range(j,len(df['Amino Acid']),3):
            if not i+1 >= len(df['Amino Acid']) and not i+2 >= len(df['Amino Acid']):
                
                threegram = df['Amino Acid'][i] + df['Amino Acid'][i+1] + df['Amino Acid'][i+2]
                threegramlist = list(threegram)
                for k in [0,1,2]:
                    if threegramlist[k].islower():
                        threegramlist[k] = 'C'
                threegram = "".join(threegramlist)   

                index = np.argwhere(code[0] == threegram) # Check to see which position a particular n-gram is
                if index.size == 0:
                    index = np.asscalar(np.argwhere(code[0]=='<unk>'))
                    print('Unknown combo:',threegram,'\nFile name:',filename)
                else:
                    index = np.asscalar(index) 
                encoded = code.iloc[index][1:].tolist()
                    
                X.append(encoded)
        total.append(X)
    if len(df['Amino Acid']) % 3 == 0: # Append 0's to the end of incomplete sequences to balance dimensionality 
        total[1].append([0]*100); total[2].append([0]*100)
    elif len(df['Amino Acid']) % 3 == 1:
        total[2].append([0]*100)
    total = np.asarray(total)
    #print(total.shape) 
    return total 

def encodeSequentialThreeGram(df, code, filename):
    X = []
    for i in range(len(df['Amino Acid'])-2):
        threegram = df['Amino Acid'][i] + df['Amino Acid'][i+1] + df['Amino Acid'][i+2]
        threegramlist = list(threegram)
        for k in [0,1,2]:
            if threegramlist[k].islower():
                threegramlist[k] = 'C'
        threegram = "".join(threegramlist)   

        index = np.argwhere(code[0] == threegram) # Check to see which position a particular n-gram is
        if index.size == 0:
            index = np.asscalar(np.argwhere(code[0]=='<unk>'))
            print('Unknown combo:',threegram,'\nFile name:',filename)
        else:
            index = np.asscalar(index) 
        encoded = code.iloc[index][1:].tolist()
        X.append(encoded)
        
    total = np.asarray(X)
    #print(total.shape) 
    return total
    
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
    
if __name__ == '__main__':
    ProcessedDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_parsed'
    OneHotDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_onehot_seq\monomeric'
    EncodedDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_threegram\monomeric\normal'
    EncodedDirectory2 = r'D:\Dropbox (MIT)\project\data\dsspdata_threegram\monomeric\sequential'
    codeFile = r'D:\Dropbox (MIT)\MIT_stuff\6.867 (Machine Learning)\Project\protVec_100d_3grams.csv'
    numFilesToParse = 1 # Number of files to parse 
    ## End User Input 
    
    # First encode the monomeric proteins 
    counter = 0 
    code = pd.read_csv(codeFile, sep= '\\t',header=None, engine='python')
    for filename in os.listdir(OneHotDirectory):
        if not os.path.isdir(os.path.join(EncodedDirectory, filename[0:4]+r'.bc')):
        
            df = pd.read_csv(os.path.join(ProcessedDirectory,filename[0:4]+r'.dssp'))
            X = threeGramEncodeAA(df, code, filename)
            X_seq = encodeSequentialThreeGram(df, code, filename) 
            
            save_array(os.path.join(EncodedDirectory, filename[0:4]+r'.bc'), X)
            save_array(os.path.join(EncodedDirectory2, filename[0:4]+r'.bc'), X_seq)
            counter += 1
            if counter % 200 == 0:
                print('Done parsing %d files...'%(int(counter)))
        # if counter == numFilesToParse:
            # break