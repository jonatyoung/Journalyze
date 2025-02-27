# Journalyze
Journalyze is a machine learning model for topic modeling in journal searches from Google Scholar in 2024 using BERTopic. 

## Group Name : Pedal Revo 🤟🏻 
Here are the people behind **Journalyze**

| No. | Name                 | NIM |
|-----|----------------------|-----|
| 1.  | Jonathan Young       |225150201111039     |
| 2.  | Federico Roberto D.S |225150200111043     |
| 3.  | Nada Firdaus                     |225150207111089     |

## Data Source ℹ️
We take journal data more precisely in 2024 using sources from https://scholar.google.com

## Directory Structure 🧱
<pre>
.
├── README.md
├── data
│   ├── scholar_results.txt
│   └── test_scrap.py
├── docker-compose.yaml
├── models
├── notebooks
│   └── eda.ipynb
├── requirements.txt
├── results
└── src
    ├── backend
    ├── frontend
    └── ml
</pre>

## Tools 🛠️
Tools we use to develop **Journalyze**

| No. | Tools                              | Function                                      |
|-----|------------------------------------|-----------------------------------------------|
| 1.  | **Beautifulsoup**                  | used for scrapping journals on google scholar |
| 2.  | **NLTK(Natural Language Toolkit)** | used to pre-process the scrapped data         |
| 3.  | **Hugging Face Transformer**       | used to embedding text                        |
| 4.  | **BERTopic**                                   | used to clustering                            |

## How to Run Our Program 🏃🏿‍♂️‍➡️


## Architecture 📐
This MLOps architecture begins with **Google Scholar** as the data source. The data is extracted using **BeautifulSoup** for web scraping, then processed with **NLTK** for text transformation. Once processed, the data is stored in a database for further use.  

For **model training and experimentation**, the stored data is used to generate **text embeddings** with **Hugging Face Transformers**. These embeddings are then clustered using **BERTopic** to identify meaningful topics.  

Once the model is trained, it is registered and versioned in **MLflow**, ensuring proper tracking of model artifacts. The stored models are then fetched by the **FastAPI backend**, which serves as an API to interact with the trained model.  

To enable **deployment and automation**, the backend is containerized using **Docker**. The **CI/CD pipeline** is responsible for pushing and committing changes to ensure smooth deployment and updates of the system.
![](./MLOps-Architecture.jpg)