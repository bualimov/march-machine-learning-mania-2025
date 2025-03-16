#!/usr/bin/env python3
# Data loader module for March Machine Learning Mania 2025

import os
import pandas as pd

# Define paths to data directories
M_DATA_DIR = 'm_csv_files'
W_DATA_DIR = 'w_csv_files'

def load_mens_data():
    """Load all men's basketball data files"""
    return load_data(gender='M')

def load_womens_data():
    """Load all women's basketball data files"""
    return load_data(gender='W')

def load_data(gender='M'):
    """
    Load all necessary data files for the specified gender.
    
    Parameters:
    -----------
    gender : str
        'M' for men's data, 'W' for women's data
    
    Returns:
    --------
    dict
        Dictionary containing all loaded dataframes
    """
    data_dir = M_DATA_DIR if gender == 'M' else W_DATA_DIR
    prefix = gender
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load teams data
    teams_file = f'{data_dir}/{prefix}Teams.csv'
    dfs['teams'] = pd.read_csv(teams_file)
    
    # Load regular season results
    reg_season_file = f'{data_dir}/{prefix}RegularSeasonCompactResults.csv'
    dfs['regular_season'] = pd.read_csv(reg_season_file)
    
    # Load detailed regular season results (available from 2003 for men, 2010 for women)
    reg_season_detailed_file = f'{data_dir}/{prefix}RegularSeasonDetailedResults.csv'
    dfs['regular_season_detailed'] = pd.read_csv(reg_season_detailed_file)
    
    # Load tournament seeds
    tourney_seeds_file = f'{data_dir}/{prefix}NCAATourneySeeds.csv'
    dfs['tourney_seeds'] = pd.read_csv(tourney_seeds_file)
    
    # Load tournament results
    tourney_results_file = f'{data_dir}/{prefix}NCAATourneyCompactResults.csv'
    dfs['tourney_results'] = pd.read_csv(tourney_results_file)
    
    # Load detailed tournament results
    tourney_detailed_file = f'{data_dir}/{prefix}NCAATourneyDetailedResults.csv'
    dfs['tourney_detailed'] = pd.read_csv(tourney_detailed_file)
    
    # Load team conferences
    team_conferences_file = f'{data_dir}/{prefix}TeamConferences.csv'
    dfs['team_conferences'] = pd.read_csv(team_conferences_file)
    
    # Load seasons data
    seasons_file = f'{data_dir}/{prefix}Seasons.csv'
    dfs['seasons'] = pd.read_csv(seasons_file)
    
    # Load Massey Ordinals (rankings) if it's men's data
    if gender == 'M':
        massey_file = f'{data_dir}/MMasseyOrdinals.csv'
        if os.path.exists(massey_file):
            dfs['massey_ordinals'] = pd.read_csv(massey_file)
    
    # Load conference tourney games
    conf_tourney_file = f'{data_dir}/{prefix}ConferenceTourneyGames.csv'
    dfs['conf_tourney_games'] = pd.read_csv(conf_tourney_file)
    
    # Load tournament slots
    tourney_slots_file = f'{data_dir}/{prefix}NCAATourneySlots.csv'
    if os.path.exists(tourney_slots_file):
        dfs['tourney_slots'] = pd.read_csv(tourney_slots_file)
    
    print(f"Loaded {len(dfs)} datasets for {gender} basketball")
    return dfs

def load_common_data():
    """Load data files that are common to both men's and women's tournaments"""
    common_dfs = {}
    
    # Load cities data
    common_dfs['cities'] = pd.read_csv('Cities.csv')
    
    # Load conferences data
    common_dfs['conferences'] = pd.read_csv('Conferences.csv')
    
    # Load sample submission file
    common_dfs['sample_submission'] = pd.read_csv('SampleSubmissionStage1.csv')
    
    print(f"Loaded {len(common_dfs)} common datasets")
    return common_dfs

def extract_seed_number(seed):
    """Extract the numeric part of the seed (e.g., 'W01' -> 1)"""
    # The seed format is like 'W01', 'X02', etc.
    # The second and third characters represent the seed number
    if isinstance(seed, str) and len(seed) >= 3:
        try:
            return int(seed[1:3])
        except ValueError:
            return None
    return None 