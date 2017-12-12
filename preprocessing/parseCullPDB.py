import os, csv 
import numpy as np

if __name__ == '__main__':
    a = []
    fileDirectory = r'D:\cullpdb.tar\pdbs'
    
    for filename in os.listdir(fileDirectory):
        a.append(filename[:4])
        
    with open('cullpdbnames.txt','w') as text_file:
        for file in a:
            text_file.write(file+'\n')
        text_file.close()