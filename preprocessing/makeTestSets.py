# Script to take out the CASP10 and CASP11 and CullPDB data set proteins 

import numpy as np
import os, shutil 
import pandas as pd 

if __name__ == '__main__':
    DataDirectory = r'D:\data'
    
    cullPDB = pd.read_csv(os.path.join(DataDirectory, 'cullpdbnames.txt'), header=None)
    cullPDB = cullPDB[0]
    casp1011 = pd.read_csv(os.path.join(DataDirectory, 'casp1011.csv'))
    casp11 = casp1011['CASP11']; casp10 = casp1011['CASP10']
    
    # for i in range(len(casp10)):
        # for filename in os.listdir(r'D:\data\dsspdata_onehot_seq\monomeric'):
            # if i < 81:
                # if casp11[i].strip() == filename[:4]:
                    # print(filename[:4],'is in CASP11')
                    # shutil.move(os.path.join(r'D:\data\dsspdata_onehot_seq\monomeric',filename),
                                # os.path.join(r'D:\data\dsspdata_onehot_seq\casp11',filename))
                    # shutil.move(os.path.join(r'D:\data\dsspdata_onehot_q3\monomeric',filename),
                                # os.path.join(r'D:\data\dsspdata_onehot_q3\casp11',filename))
                    # shutil.move(os.path.join(r'D:\data\dsspdata_onehot_q8\monomeric',filename),
                                # os.path.join(r'D:\data\dsspdata_onehot_q8\casp11',filename))
                    # shutil.move(os.path.join(r'D:\data\dsspdata_threegram\monomeric\sequential',filename),
                                # os.path.join(r'D:\data\dsspdata_threegram\casp11',filename))
            # if casp10[i].strip() == filename[:4]:
                # print(filename[:4], 'is in CASP10')
                # shutil.move(os.path.join(r'D:\data\dsspdata_onehot_seq\monomeric',filename),
                            # os.path.join(r'D:\data\dsspdata_onehot_seq\casp10',filename))
                # shutil.move(os.path.join(r'D:\data\dsspdata_onehot_q3\monomeric',filename),
                            # os.path.join(r'D:\data\dsspdata_onehot_q3\casp10',filename))
                # shutil.move(os.path.join(r'D:\data\dsspdata_onehot_q8\monomeric',filename),
                            # os.path.join(r'D:\data\dsspdata_onehot_q8\casp10',filename))
                # shutil.move(os.path.join(r'D:\data\dsspdata_threegram\monomeric\sequential',filename),
                            # os.path.join(r'D:\data\dsspdata_threegram\casp10',filename))
    
    for i in range(len(cullPDB)):
        for filename in os.listdir(r'D:\data\dsspdata_onehot_seq\monomeric'):
            if cullPDB[i].strip() == filename[:4]:
                print(filename[:4], 'is in CullPDB')
                shutil.move(os.path.join(r'D:\data\dsspdata_onehot_seq\monomeric',filename),
                            os.path.join(r'D:\data\dsspdata_onehot_seq\cullpdb',filename))
                shutil.move(os.path.join(r'D:\data\dsspdata_onehot_q3\monomeric',filename),
                            os.path.join(r'D:\data\dsspdata_onehot_q3\cullpdb',filename))
                shutil.move(os.path.join(r'D:\data\dsspdata_onehot_q8\monomeric',filename),
                            os.path.join(r'D:\data\dsspdata_onehot_q8\cullpdb',filename))
                shutil.move(os.path.join(r'D:\data\dsspdata_threegram\monomeric\sequential',filename),
                            os.path.join(r'D:\data\dsspdata_threegram\cullpdb',filename))