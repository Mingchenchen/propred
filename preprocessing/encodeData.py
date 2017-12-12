import numpy as np
import pandas as pd 
import os, glob, bcolz 
from parseData import * 
import multiprocessing as mp 

def oneHotEncodeAA(df, filename):
    # Given a parsed dataframe, one hot encode the amino acid sequence as a vector 
    ## Positions: A = 0, R = 1, N = 2, D = 3, V = 4, C = 5, E = 6, Q = 7, Y = 8, G = 9, H = 10, I = 11, 
    ##            L = 12, K = 13, M = 14, F = 15, P = 16, S = 17, T = 18, W = 19, X = 20, B = 21, Z = 22, ! = 23 [end] (X denotes unknown amino acids)
    X = []
    for aa in df['Amino Acid']:
        x = 24 * [0]
        if aa == 'A': x[0] = 1
        elif aa == 'R': x[1] = 1
        elif aa == 'N': x[2] = 1
        elif aa == 'D': x[3] = 1
        elif aa == 'V': x[4] = 1
        elif aa == 'C': x[5] = 1
        elif aa == 'E': x[6] = 1
        elif aa == 'Q': x[7] = 1
        elif aa == 'Y': x[8] = 1
        elif aa == 'G': x[9] = 1
        elif aa == 'H': x[10] = 1
        elif aa == 'I': x[11] = 1
        elif aa == 'L': x[12] = 1
        elif aa == 'K': x[13] = 1
        elif aa == 'M': x[14] = 1
        elif aa == 'F': x[15] = 1
        elif aa == 'P': x[16] = 1
        elif aa == 'S': x[17] = 1
        elif aa == 'T': x[18] = 1
        elif aa == 'W': x[19] = 1
        elif aa == 'X': x[20] = 1
        elif aa == 'B': x[21] = 1
        elif aa == 'Z': x[22] = 1
        elif aa == '!': x[23] = 1
        elif aa.islower(): x[5] = 1 # lower case letteres denote disulfide bridged cysteines 
        else: print('Unknown residue:',aa,'\nFile name:',filename)
        X.append(x)
    return np.asarray(X) 
    
def oneHotEncodeSecStruc(df, filename):
    # Given dataframe, one hot encode the 8 and 3 secondary structure possibilities 
    ## H = 0, B = 1, E = 2, G = 3, I = 4, T = 5, S = 6, L = 7 
    ## Q3 - G,I,H = 0; B,E = 1; T,S,L = 2 
    q3 = []; q8 = []
    for ss in df['Secondary Structure']:
        x3 = 3*[0]; x8 = 8*[0]
        if ss == 'H': x3[0] = 1; x8[0] = 1
        elif ss == 'B': x3[1] = 1; x8[1] = 1
        elif ss == 'E': x3[1] = 1; x8[2] = 1
        elif ss == 'G': x3[0] = 1; x8[3] = 1
        elif ss == 'I': x3[0] = 1; x8[4] = 1
        elif ss == 'T': x3[2] = 1; x8[5] = 1
        elif ss == 'S': x3[2] = 1; x8[6] = 1
        elif ss == 'L': x3[2] = 1; x8[7] = 1
        else: print('Unknown secondary structure:',ss, '\nFile name:',filename)
        q3.append(x3); q8.append(x8) 
        
    return np.asarray(q3), np.asarray(q8) 
    
def encodeTertStruc(df):
    pass 
    
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
    
if __name__ == '__main__':
    ProcessedDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_parsed'
    EncodedDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_onehot'
    numFilesToParse = 1 # Number of files to parse 
    ## End User Input 

    counter = 0 
    for filename in os.listdir(ProcessedDirectory):
        if not os.path.isdir(os.path.join(EncodedDirectory+r'_seq', filename[0:4]+r'.bc')):
            df = pd.read_csv(os.path.join(ProcessedDirectory,filename))
            X = oneHotEncodeAA(df, filename)
            q3, q8 = oneHotEncodeSecStruc(df, filename)
            
            save_array(os.path.join(EncodedDirectory+r'_seq', filename[0:4]+r'.bc'), X)
            save_array(os.path.join(EncodedDirectory+r'_q3', filename[0:4]+r'.bc'), q3)
            save_array(os.path.join(EncodedDirectory+r'_q8', filename[0:4]+r'.bc'), q8)
            counter += 1
            if counter % 200 == 0:
                print('Done parsing %d files...'%(int(counter)))
        # if counter == numFilesToParse:
            # break