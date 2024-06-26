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
    training_file = config["paths"]["training_file"]
    return preprocessed_dataset_dir, training_file

preprocessed_dataset_dir, training_file = read_ini("/s/chopin/k/grad/mbrad/cs535/EEG_Classification/EEG-DICE-net/feature_extraction/config.ini")

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
        Class="Missing" # (naming is sub-088), see https://openneuro.org/datasets/ds004504/versions/1.0.7/file-display/participants.tsv for labels
        subj=band_subj.split("-")[-1]
        subjInt = int(subj)
        if subjInt < 37:
            Class = "A"
        elif subjInt < 66:
            Class = "C"
        else:
            Class = "F"
        conn=np.load(conn_file)
        list_conn=[s for s in np.load(conn_file)]
        list_band=[s for s in np.load(band_file)]
        full_conn = []
        full_band = []
        for j,conn in enumerate(list_conn):
            full_conn.extend(conn) #for one long list 
            full_band.extend(list_band[j])
        d={'subj': subj, 'conn':full_conn,'band':full_band,'class':Class}
        training_list.append(d)
    training_dataframe = pd.DataFrame(training_list)
    return training_dataframe
    
if __name__ == "__main__":
    print(preprocessed_dataset_dir)
    preprocessed_dataset = find_files(preprocessed_dataset_dir, ".set")
    preprocessed_dataset.sort()
    for set_file in preprocessed_dataset:
        print(set_file)
    create_numpy_connAndband_files(preprocessed_dataset)
    created_nmpy_files = find_files(preprocessed_dataset_dir, ".npy")
    for nmpy_name in created_nmpy_files:
        print(nmpy_name)
    training_dataframe = create_training_dataset(created_nmpy_files)
    training_dataframe.to_pickle(training_file)