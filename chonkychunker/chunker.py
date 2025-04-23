import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class TextChunker:
    def __init__(self, metric='euclidean', top_k=5, distance_threshold=2):
        self.metric = metric
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.texts = None

    def embed(self, texts):
        self.texts = texts
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        if self.metric == 'cosine':
            self.embeddings = normalize(self.embeddings, norm='l2')

        return self.embeddings

    def cluster(self, return_embeddings=False):
        if self.metric == 'cosine':
            nn = NearestNeighbors(n_neighbors=self.top_k, metric='cosine')
            nn.fit(self.embeddings)
            distances, indices = nn.kneighbors(self.embeddings)
        else:
            tree = BallTree(self.embeddings, metric=self.metric)

        clusters = []
        visited = set()
        for idx in range(len(self.embeddings)):
            if idx in visited:
                continue

            if self.metric == 'cosine':
                neighbors = indices[idx]
                dists = distances[idx]
            else:
                dists, neighbors = tree.query([self.embeddings[idx]], k=self.top_k)
                neighbors = neighbors[0]
                dists = dists[0]

            if self.distance_threshold is not None:
                neighbors = [n for n, d in zip(neighbors, dists) if d <= self.distance_threshold]

            group = set(neighbors)
            clusters.append([self.texts[i] for i in group])
            visited.update(group)

        unique_clusters = self.remove_duplicates(clusters)
        if return_embeddings:
            return unique_clusters, self.embeddings
        return unique_clusters

    def remove_duplicates(self, clusters):
        seen = set()
        unique_clusters = []
        for cluster in clusters:
            unique_cluster = []
            for item in cluster:
                if item not in seen:
                    unique_cluster.append(item)
                    seen.add(item)
            if unique_cluster:
                unique_clusters.append(unique_cluster)
        return unique_clusters

    def get_vector_output(self):
        output = []
        clusters = self.cluster()
        for cluster_id, cluster in enumerate(clusters):
            for text in cluster:
                idx = self.texts.index(text)
                output.append({
                    "id": f"doc_{idx}",
                    "text": text,
                    "embedding": self.embeddings[idx].tolist(),
                    "cluster": cluster_id
                })
        return output

    def to_langchain_documents(self):
        clusters = self.cluster()
        documents = []
        for cluster_id, cluster in enumerate(clusters):
            for text in cluster:
                documents.append(Document(page_content=text, metadata={"cluster": cluster_id}))
        return documents

