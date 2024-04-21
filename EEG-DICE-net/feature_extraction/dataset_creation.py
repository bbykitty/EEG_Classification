# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:40:50 2022

@author: AndreasMiltiadous
"""
import pandas as pd
import numpy as np
import global_spectral_coherence_computation as gs
import relative_band_power_computation as rb
import split_dataset as sp
import os
import configparser

def read_ini(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    preprocessed_dataset_dir = config["paths"]["preprocessed_dataset_dir"]
    feature_dir = config["paths"]["feature_dir"]
    training_dir = config["paths"]["training_dir"]
    validation_dir = config["paths"]["validation_dir"]
    testing_dir = config["paths"]["testing_dir"]
    return preprocessed_dataset_dir, feature_dir, training_dir, validation_dir, testing_dir

preprocessed_dataset_dir, feature_dir, training_dir, validation_dir, testing_dir = read_ini("/s/chopin/k/grad/mbrad/cs535/EEG_Classification/EEG-DICE-net/feature_extraction/config.ini")

def find_files(directory_path, filetype):
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(filetype):
                all_files.append(os.path.join(root, file))
    return all_files

def create_numpy_connAndband_files(filelist=None):

    if filelist==None:
        print("Dataset list is empty!")
        return
    print("Creating feature set!")
    subject_list, filenames=gs.create_subject_epoch_list(filelist,30)
    conn_results=[gs.calc_spectral_coherence_connectivity(subject) for subject in subject_list]
    band_results=[rb.calc_relative_band_power(subject) for subject in subject_list]
    for i,result in enumerate(conn_results):
        listt=sp.split_array_to_fixed_sized_arrays(result,splitsize=30,to_csv=False,filename=filenames[i])
        arr_coherence = np.array(listt)
        
        ##naming
        filepath=os.path.split(filenames[i])[0]
        name=os.path.split(filenames[i])[1]
        plain_name=os.path.splitext(name)[0]
        with open( filepath + '/' + plain_name + "_conn.npy", 'wb') as f:
            print("Saving " + filepath + '/' + plain_name + "_conn.npy")
            np.save(f, arr_coherence)
    
    for i,result in enumerate(band_results):
        listt=sp.split_array_to_fixed_sized_arrays(result,splitsize=30,to_csv=False,filename=filenames[i])
        arr_band = np.array(listt)
        
        ##naming
        filepath=os.path.split(filenames[i])[0]
        name=os.path.split(filenames[i])[1]
        plain_name=os.path.splitext(name)[0]
        with open( filepath + '/' + plain_name + "_band.npy", 'wb') as f:
            print("Saving " + filepath + '/' + plain_name + "_band.npy")
            np.save(f, arr_band)
        
def create_training_dataset(filelist):
    '''
    Gets (or asks the user for) a list of .npy files that MUST be named as A4_band.npy and A4_conn.npy (S0S: ordered alphabetically)
    returns training_dataframe with columns [subj, conn (numpy array), band (numpy array), class]

    Parameters
    ----------
    filelist : TYPE, list of filenames
        DESCRIPTION. The default is None.

    Returns
    -------
    training_dataframe : dataframe, each row is one training sample.

    '''
    if len(filelist)==0:
        print("Training directory is empty!")
        return
    filelist.sort()
    band_list=[]
    conn_list=[]
    training_list = []
    training_dataframe=pd.DataFrame(columns=['subj','conn','band','class'])    
    for file in filelist:
        path=os.path.split(file)[0]
        name=os.path.split(file)[1]
        plain_name=os.path.splitext(name)[0]
        if plain_name.split("_")[-1]=="band":
            band_list.append(file)
        elif plain_name.split("_")[-1]=="conn":
            conn_list.append(file)
        else:
            print("something wrong: ")
            print(plain_name.split("_")[-1])
            #return -1;
    for i,band_file in enumerate(band_list):
        conn_file=conn_list[i]
        band_subj=os.path.splitext(os.path.split(band_file)[1])[0].split("_")[0]
        conn_subj=os.path.splitext(os.path.split(conn_file)[1])[0].split("_")[0]
        print(band_subj,conn_subj)
        if conn_subj!=band_subj:
            print("error in file selection")
            return
        Class=band_subj[0]
        subj=band_subj[1:]
        conn=np.load(conn_file)
        list_conn=[s for s in np.load(conn_file)]
        list_band=[s for s in np.load(band_file)]
        for j,conn in enumerate(list_conn):
            band=list_band[j]
            d={'subj': subj, 'conn':conn,'band':band,'class':Class}
            # ser=pd.Series(data=d,index=['subj','conn','band','class'])
            training_list.append(d)#todo
        training_dataframe = pd.DataFrame(training_list)
    return training_dataframe
    
if __name__ == "__main__":
    print(preprocessed_dataset_dir)
    preprocessed_dataset = find_files(preprocessed_dataset_dir, ".set")
    # preprocessed_dataset.sort()
    # for set_file in preprocessed_dataset:
    #     print(set_file)
    # preprocessed_sample = preprocessed_dataset[0:3]
    # create_numpy_connAndband_files(preprocessed_dataset)
    created_nmpy_files = find_files(preprocessed_dataset_dir, ".npy")
    for nmpy_name in created_nmpy_files:
        print(nmpy_name)
    training_dataframe = create_training_dataset(created_nmpy_files)
    training_dataframe.to_pickle(training_dir+'/TrainingDataset.pkl')
    # #validation
    # validation_dataframe = all_dataframes[1]
    # validation_dataframe.to_pickle(validation_dir+'/ValidationDataset.pkl')
    # #testing
    # testing_dataframe = all_dataframes[2]
    # testing_dataframe.to_pickle(testing_dir+'/TestingDataset.pkl')