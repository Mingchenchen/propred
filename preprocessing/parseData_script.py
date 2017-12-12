import numpy as np
import pandas as pd 
import os, glob, bcolz 
from parseData import * 
import multiprocessing as mp 

# Loop over some set of files and make them into more useable files 

    
if __name__ == '__main__':
    RawDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_raw'
    InterimDirectory = r'D:\Dropbox (MIT)\project\data\dsspdata_parsed'
    numFilesToParse = 1 # Number of files to parse 
    ## End User Input 
    
    
    counter = 0 
    for filename in os.listdir(RawDirectory):
        if not os.path.isfile(os.path.join(InterimDirectory, filename)):
            dsspFile = DSSPData()
            dsspFile.parseDSSP(os.path.join(RawDirectory, filename))
            AAs = dsspFile.getAAs()
            SSs = dsspFile.getSecStruc()
            xCa = dsspFile.getX(); yCa = dsspFile.getY(); zCa = dsspFile.getZ()
            TCO = dsspFile.getTCO()
            kappa = dsspFile.getKAPPA()
            alpha = dsspFile.getALPHA()
            phi = dsspFile.getPHI()
            psi = dsspFile.getPSI()
            
            data = {'Amino Acid':AAs, 'Secondary Structure':SSs, 'TCO':TCO, 'Kappa':kappa,'Alpha':alpha,'Phi':phi,'Psi':psi,'X':xCa,'Y':yCa,'Z':zCa}
            df = pd.DataFrame(data=data)
            df.to_csv(os.path.join(InterimDirectory,filename))
            counter += 1
            if counter % 200 == 0:
                print('Done parsing %d files...'%(int(counter)))
        # if counter == numFilesToParse:
            # break