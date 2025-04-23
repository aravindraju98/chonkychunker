import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from transformers import AutoTokenizer

class TextChunker:
    """
    A Ball Tree-based semantic text chunker using SentenceTransformers for embeddings.
    Supports cosine and Euclidean distance, and outputs LangChain-compatible Documents
    and vector DB ingestion formats.
    """
    def __init__(self, metric='euclidean', top_k=5, distance_threshold=2, max_tokens=None):
        self.metric = metric
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.texts = None
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') if max_tokens else None

    def embed(self, texts):
        """
        Embed a list of texts using SentenceTransformer.

        Args:
            texts (List[str]): A list of input text strings.

        Returns:
            np.ndarray: Array of embeddings.
        """
        self.texts = texts
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        if self.metric == 'cosine':
            self.embeddings = normalize(self.embeddings, norm='l2')

        return self.embeddings

    def cluster(self, return_embeddings=False):
        """
        Cluster texts using BallTree or NearestNeighbors.

        Args:
            return_embeddings (bool): Whether to return embeddings with clusters.

        Returns:
            List[List[str]]: A list of clusters with grouped texts.
        """
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
        """
        Remove duplicate entries across clusters.

        Args:
            clusters (List[List[str]]): Clustered text groups.

        Returns:
            List[List[str]]: Deduplicated clusters.
        """
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

    def truncate_by_tokens(self, texts):
        if not self.max_tokens:
            return "\n".join(texts)

        current_tokens = 0
        selected_texts = []
        for text in texts:
            tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            if current_tokens + tokens > self.max_tokens:
                break
            selected_texts.append(text)
            current_tokens += tokens
        return "\n".join(selected_texts)

    def get_vector_output(self, merge=False):
        """
        Export clustered text and embeddings for vector DB ingestion.

        Args:
            merge (bool): If True, merge texts per cluster into one document.

        Returns:
            List[Dict]: List of entries with id, text, embedding, and cluster ID.
        """
        output = []
        clusters = self.cluster()
        if merge:
            for cluster_id, cluster in enumerate(clusters):
                merged_text = self.truncate_by_tokens(cluster)
                embeddings = [self.embeddings[self.texts.index(t)] for t in cluster]
                avg_embedding = np.mean(embeddings, axis=0).tolist()
                output.append({
                    "id": f"cluster_{cluster_id}",
                    "text": merged_text,
                    "embedding": avg_embedding,
                    "cluster": cluster_id
                })
        else:
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

    def to_langchain_documents(self, merge=False):
        """
        Convert clusters to LangChain-compatible Document objects.

        Args:
            merge (bool): If True, merge texts in each cluster into one Document.

        Returns:
            List[Document]: LangChain Documents with cluster metadata.
        """
        clusters = self.cluster()
        documents = []
        if merge:
            for cluster_id, cluster in enumerate(clusters):
                merged_text = self.truncate_by_tokens(cluster)
                documents.append(Document(page_content=merged_text, metadata={"cluster": cluster_id}))
        else:
            for cluster_id, cluster in enumerate(clusters):
                for text in cluster:
                    documents.append(Document(page_content=text, metadata={"cluster": cluster_id}))
        return documents