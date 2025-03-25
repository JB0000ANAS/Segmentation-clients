import os
os.environ['LOKY_MAX_CPU_COUNT'] = '14'

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.spatial.distance import cdist
import torch
from transformers import GPT2Model, GPT2Tokenizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy
import sqlite3
import pickle
from joblib import Parallel, delayed
import plotly.utils
import json
 


# Constants for cache files
EMBEDDINGS_CACHE_FILE = "embeddings.pkl"
BERT_EMBEDDINGS_CACHE_FILE = "bert_embeddings.pkl"

def process_data(input_data):
    results = Parallel(n_jobs=-1)(delayed(compute_function)(data) for data in input_data)
    return results

def compute_function(data):
    return data * data

app = Flask(__name__, template_folder='../templates', static_folder='frontend')
CORS(app)

# Global variables
nlp = None
gpt_model = None
tokenizer = None
embeddings_cache = None
bert_embeddings_cache = None
df = None
sentence_model = Nones
topic_model = None

def get_db_connection():
    import spacy
    # Charger le modèle dans chaque worker (ceci peut ralentir si le modèle est volumineux)
    nlp_worker = spacy.load('fr_core_news_sm')
    database_path = "C:/Users/ajebali/OneDrive - N.D.K/Bureau/alt/S_2_A/tt/projet_clustering/data/nvv_database.db"
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn

def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop and not token.like_num])

def remove_stopwords(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

def load_data_from_sqlite():
    global df
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        
        test_query = "SELECT * FROM rpv_txt LIMIT 1"
        test_df = pd.read_sql_query(test_query, conn)
        print("Colonnes disponibles dans la table:", test_df.columns.tolist())
        
        query = "SELECT TEXTE_CLEAN FROM rpv_txt"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No data was loaded from the database")
            return None
            
        if 'TEXTE_CLEAN' not in df.columns and 'TEXTE_CLEAN' in df.columns:
            df = df.rename(columns={'TEXTE_CLEAN': 'TEXTE_CLEAN'})
        
        # Sample the data
        df = df.sample(n=5000, random_state=42)
        
        # Clean and preprocess the text
        df['TEXTE_CLEAN'] = df['TEXTE_CLEAN'].fillna('').apply(clean_and_lemmatize)
            
        return df
        
    except Exception as e:
        print(f"Error loading data from SQLite: {e}")
        return None
    finally:
        if conn:
            conn.close()

def init_models():
    global nlp, gpt_model, tokenizer, sentence_model, topic_model
    try:
        nlp = spacy.load('fr_core_news_sm')
        path_to_model = "C:\\Users\\ajebali\\gpt2-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(path_to_model, local_files_only=True)
        gpt_model = GPT2Model.from_pretrained(path_to_model, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize BERTopic models
        sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        topic_model = BERTopic(
            embedding_model=sentence_model,
            language='multilingual',
            calculate_probabilities=True,
            min_topic_size=10,
            n_gram_range=(1, 2)
            
        )
        return True
    except Exception as e:
        print(f"Erreur lors de l'initialisation des modèles: {e}")
        return False

def reduire_et_clusterer(document_embeddings, n_clusters=3):
    try:
        if document_embeddings is None or np.isnan(document_embeddings).any():
            raise ValueError("Invalid embeddings: contains NaN values")

        if len(document_embeddings.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {document_embeddings.shape}")

        document_embeddings = document_embeddings.astype(np.float64)

        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(document_embeddings)

        if np.isnan(reduced_embeddings).any():
            raise ValueError("PCA resulted in NaN values")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)

        return reduced_embeddings, clusters

    except Exception as e:
        print(f"Error in reduire_et_clusterer: {e}")
        raise

def generer_embeddings(docs, model, tokenizer, max_length=128, batch_size=8):
    global embeddings_cache

    try:
        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                embeddings_cache = pickle.load(f)
            print("Embeddings chargés depuis le cache.")
            return embeddings_cache

        if not docs or len(docs) == 0:
            raise ValueError("Empty document list")

        all_embeddings = []
        num_docs = len(docs)
        
        for i in range(0, num_docs, batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_docs = [doc if isinstance(doc, str) else "" for doc in batch_docs]
            
            inputs = tokenizer(batch_docs, return_tensors='pt', max_length=max_length, 
                             truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            if np.isnan(embeddings).any():
                raise ValueError("Model generated NaN embeddings")
                
            all_embeddings.append(embeddings)
        
        embeddings_cache = np.vstack(all_embeddings)
        
        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(embeddings_cache, f)
        print("Embeddings enregistrés dans le cache.")

        return embeddings_cache
       
    except Exception as e:
        print(f"Error in generer_embeddings: {e}")
        raise

def visualiser_clusters_json(reduced_embeddings, clusters, texts):
    cluster_data = []
    for i in range(len(reduced_embeddings)):
        cluster_data.append({
            'x': float(reduced_embeddings[i][0]),
            'y': float(reduced_embeddings[i][1]),
            'cluster': int(clusters[i]),
            'text': texts.iloc[i]
        })
    return cluster_data

def calculate_elbow_data(embeddings, max_clusters=10):
    inertias = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        inertias.append({
            'clusters': k,
            'inertia': float(kmeans.inertia_)
        })
    return inertias

def calculate_validation_metrics(embeddings, clusters):
    """Calculate multiple cluster validation metrics."""
    try:
        metrics = {
            'silhouette': float(silhouette_score(embeddings, clusters)),
            'calinski': float(calinski_harabasz_score(embeddings, clusters)),
            'davies': float(davies_bouldin_score(embeddings, clusters))
        }
        return metrics
    except Exception as e:
        print(f"Error calculating validation metrics: {e}")
        return None

def calculate_gap_statistic(data, k_max=10, n_references=5):
    """
    Calculate Gap statistic for clustering validation
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    reference_inertias = []
    for k in range(1, k_max + 1):
        local_inertias = []
        for _ in range(n_references):
            # Generate reference dataset
            reference = np.random.uniform(
                data.min(axis=0), data.max(axis=0), data.shape)
            
            # Fit k-means to reference dataset
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(reference)
            local_inertias.append(kmeans.inertia_)
        
        reference_inertias.append(np.mean(local_inertias))

    # Calculate gap statistic
    gaps = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(data)
        gaps.append(np.log(reference_inertias[k-1]) - np.log(kmeans.inertia_))
    
    return gaps

def perform_hierarchical_clustering(embeddings, n_clusters):
    """Perform hierarchical clustering and return linkage matrix."""
    linkage_matrix = linkage(embeddings, method='ward')
    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    return linkage_matrix, clusters

def perform_dbscan_clustering(embeddings):
    """Perform DBSCAN clustering."""
    eps = 0.5  # You might want to make this parameter adjustable
    min_samples = 5
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(embeddings)
    return clusters    

@app.before_request
def initialize():
    global df, nlp, gpt_model, tokenizer, embeddings_cache, sentence_model, topic_model

    if df is None:
        try:
            print("Starting initialization...")
            df = load_data_from_sqlite()
            
            if df is None or df.empty:
                raise ValueError("Failed to load data from SQLite")

            if nlp is None or gpt_model is None or tokenizer is None:
                if not init_models():
                    raise ValueError("Failed to initialize models")

            embeddings_cache = generer_embeddings(df['TEXTE_CLEAN'].tolist(), gpt_model, tokenizer)

        except Exception as e:
            print(f"Initialization error: {e}")
            return jsonify({'error': 'Initialization failed'}), 500

@app.route('/')
def index():
    return render_template('index.html')


# Add this to your existing Flask routes:
@app.route('/api/cluster-count', methods=['GET'])
def get_cluster_info():
    global embeddings_cache
    try:
        if embeddings_cache is None:
            return jsonify({'error': 'No embeddings available'}), 500
            
        # Calculate the maximum reasonable number of clusters
        max_clusters = min(20, len(embeddings_cache) // 5)  # Or another reasonable limit
        
        return jsonify({
            'max_clusters': max_clusters,
            'success': True
        })
    except Exception as e:
        print(f"Error in get_cluster_info: {e}")
        return jsonify({'error': str(e)}), 500
    
    

@app.route('/api/clustering-data', methods=['GET'])
def get_clustering_data():
    global df, embeddings_cache

    try:
        if df is None or embeddings_cache is None:
            return jsonify({'error': 'Data not initialized'}), 500

        clustering_method = request.args.get('method', 'kmeans')
        n_clusters = min(max(int(request.args.get('n_clusters', 3)), 2), 20)

        # Perform dimensionality reduction
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_cache)

        # Perform clustering based on selected method
        if clustering_method == 'kmeans':
            _, clusters = reduire_et_clusterer(embeddings_cache, n_clusters)
        elif clustering_method == 'hierarchical':
            linkage_matrix, clusters = perform_hierarchical_clustering(embeddings_cache, n_clusters)
        elif clustering_method == 'dbscan':
            clusters = perform_dbscan_clustering(embeddings_cache)
        else:
            return jsonify({'error': 'Invalid clustering method'}), 400

        # Calculate validation metrics
        validation_metrics = calculate_validation_metrics(embeddings_cache, clusters)
        gap_stats = calculate_gap_statistic(embeddings_cache)
        
        # Generate cluster visualization data
        cluster_data = visualiser_clusters_json(reduced_embeddings, clusters, df['TEXTE_CLEAN'])
        elbow_data = calculate_elbow_data(embeddings_cache)

        # Calculate cluster statistics
        unique_clusters = np.unique(clusters)
        cluster_stats = []
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_size = np.sum(mask)
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(clusters) * 100),
                'sample_texts': df['TEXTE_CLEAN'][mask].head(3).tolist()
            })

        return jsonify({
            'clusterData': cluster_data,
            'elbowData': elbow_data,
            'validationMetrics': validation_metrics,
            'gapStatistic': gap_stats,
            'clusterStats': cluster_stats,
            'success': True
        })

    except Exception as e:
        print(f"Error in get_clustering_data: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/update-clusters', methods=['GET'])
def update_clusters():
    global df, embeddings_cache

    try:
        if df is None or embeddings_cache is None:
            return jsonify({'error': 'Data not initialized'}), 500

        n_clusters = int(request.args.get('n', 3))
        reduced_embeddings, clusters = reduire_et_clusterer(embeddings_cache, n_clusters)
        cluster_data = visualiser_clusters_json(reduced_embeddings, clusters, df['TEXTE_CLEAN'])

        return jsonify({
            'clusterData': cluster_data,
            'success': True
        })

    except Exception as e:
        print(f"Error in update_clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Modify the Flask route to accept n_topics parameter
@app.route('/api/topic-analysis', methods=['GET'])
def get_topic_analysis():
    global df, topic_model, sentence_model
    
    try:
        if df is None or df.empty:
            return jsonify({'error': 'No documents available'}), 500

        # Récupérer le nombre de topics souhaité depuis la requête
        # Par défaut, n_topics vaut 10. On limite ensuite entre 1 et 20.
        n_topics = min(max(int(request.args.get('n_topics', 10)), 1), 20)
        
        # Réinitialiser le modèle BERTopic en passant le paramètre nr_topics
        topic_model = BERTopic(
            embedding_model=sentence_model,
            language='multilingual',
            calculate_probabilities=True,
            min_topic_size=5,      # Vous pouvez ajuster ce paramètre selon vos données
            n_gram_range=(1, 2),
            nr_topics=n_topics     # Ici, on fixe le nombre de topics souhaité
        )

        # Appliquer fit_transform sur la colonne nettoyée des textes
        topics, probs = topic_model.fit_transform(df['TEXTE_CLEAN'])
        
        # Récupérer les informations sur les topics sous forme de DataFrame
        topic_info = topic_model.get_topic_info()
        topics_data = topic_info.to_dict('records')
        
        # Générer les visualisations
        intertopic_map = topic_model.visualize_topics()
        barchart = topic_model.visualize_barchart(top_n_topics=min(n_topics, 10), n_words=5)
        hierarchy = topic_model.visualize_hierarchy()
        heatmap = topic_model.visualize_heatmap()
        
        # Convertir les figures Plotly en JSON pour renvoyer au client
        barchart_json = json.loads(json.dumps(barchart, cls=plotly.utils.PlotlyJSONEncoder))
        hierarchy_json = json.loads(json.dumps(hierarchy, cls=plotly.utils.PlotlyJSONEncoder))
        heatmap_json = json.loads(json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder))
        intertopic_map_json = json.loads(json.dumps(intertopic_map, cls=plotly.utils.PlotlyJSONEncoder))
        
        # Récupérer quelques documents représentatifs par topic
        topic_docs = {}
        for topic_id in topic_info[topic_info['Topic'] != -1]['Topic']:
            topic_docs[str(topic_id)] = topic_model.get_representative_docs(topic_id)[:3]
        
        return jsonify({
            'topics_data': topics_data,
            'topic_docs': topic_docs,
            'visualizations': {
                'barchart': barchart_json,
                'hierarchy': hierarchy_json,
                'heatmap': heatmap_json,
                'intertopic_map': intertopic_map_json
            },
            'success': True
        })

    except Exception as e:
        print(f"Error in topic analysis: {str(e)}")
        return jsonify({
            'error': 'An error occurred during topic analysis',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    if not init_models():
        print("Error: Models could not be initialized. Exiting.")
        exit()
    app.run(debug=True, port=5000)