## Organize data into multimeric and monomeric proteins 

import numpy as np
import pandas as pd 
import os, linecache, shutil 

def sortMultiMonomeric(filename):
    num_chains = int(linecache.getline(filename, 7)[6:8].strip())
    isMultimeric = False 
    if num_chains > 1:
        isMultimeric = True 
    elif num_chains == 0:
        print(filename,"got 0 chains")
        print("What the files says",linecache.getline(filename, 7)[:14])
        isMultimeric = True 
        #raise RuntimeError('BUG ALERT, we got 0 chains')
    return isMultimeric 
        

if __name__ == "__main__":
    ## Search through files to see if "!" (i.e. end of a sequence)
    EncodedDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_onehot'
    RawDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_raw'
    numFilesToParse = 1 # Number of files to parse
    ## END user Input 

    counter = 0 
    for filename in os.listdir(RawDirectory):
        if not os.path.isdir(os.path.join(EncodedDirectory+r'_seq'+r'\multimeric', filename[0:4]+r'.bc')) and not os.path.isdir(os.path.join(EncodedDirectory+r'_seq'+r'\monomeric', filename[0:4]+r'.bc')):
            isMultimeric = sortMultiMonomeric(os.path.join(RawDirectory,filename))
            if isMultimeric:
                shutil.move(os.path.join(EncodedDirectory+r'_seq', filename[0:4]+r'.bc'), 
                            os.path.join(EncodedDirectory+r'_seq'+r'\multimeric',filename[0:4]+r'.bc'))
                shutil.move(os.path.join(EncodedDirectory+r'_q3', filename[0:4]+r'.bc'), 
                            os.path.join(EncodedDirectory+r'_q3'+r'\multimeric',filename[0:4]+r'.bc'))
                shutil.move(os.path.join(EncodedDirectory+r'_q8', filename[0:4]+r'.bc'), 
                            os.path.join(EncodedDirectory+r'_q8'+r'\multimeric',filename[0:4]+r'.bc'))
            else:
                shutil.move(os.path.join(EncodedDirectory+r'_seq', filename[0:4]+r'.bc'), 
                            os.path.join(EncodedDirectory+r'_seq'+r'\monomeric',filename[0:4]+r'.bc'))
                shutil.move(os.path.join(EncodedDirectory+r'_q3', filename[0:4]+r'.bc'), 
                            os.path.join(EncodedDirectory+r'_q3'+r'\monomeric',filename[0:4]+r'.bc'))
                shutil.move(os.path.join(EncodedDirectory+r'_q8', filename[0:4]+r'.bc'), 
                            os.path.join(EncodedDirectory+r'_q8'+r'\monomeric',filename[0:4]+r'.bc'))

            counter += 1
            if counter % 200 == 0:
                print('Done parsing %d files...'%(int(counter)))
        # if counter == numFilesToParse:
            # break
    