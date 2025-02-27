# Final Project - Machine Learning Operations
This repository contains the final project of the Machine learning operations course. This project develops a machine learning model for topic modeling in journal searches from Google Scholar in 2024 using BERTopic. 

## Group Name : Pedal Revo 

| No. | Name                 | NIM |
|-----|----------------------|-----|
| 1.  | Jonathan Young       |225150201111039     |
| 2.  | Federico Roberto D.S |225150200111043     |
| 3.  | Nada Firdaus                     |225150207111089     |

## Data Source
Resource =>  https://scholar.google.com

## Directory Structure
<pre>
. 
├── README.md
├── scholar_results.txt
└── test_scrap.py </pre>

## Tools
1. **Beautifulsoup** -> used for scrapping journals on google scholar
2. **NLTK(Natural Language Toolkit)** -> used to pre-process the scrapped data

## Architecture
The architectural drawing below explains the flow of how the process of retrieving data from Google Scholar using the beautiful soup scrapping tool and pre-processing the scrapped data and then the data is stored in a txt file.

![](./MLOps-Architecture.jpg)