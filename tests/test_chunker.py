import unittest
from chonkychunker import TextChunker

class TestTextChunker(unittest.TestCase):

    def setUp(self):
        self.texts = [
            "the milk is spoiled",
            "themonuclear physics is easy",
            "the Egg is boiled",
            "water is added to cooldown",
            "Salts are formed when an acid reacts with base",
            "Salt is added for taste",
            "car is washed",
            "detergent is good for removing stains",
            "macdonalds meal contains french fries"
        ]
        self.chunker = TextChunker(metric='cosine', top_k=4, distance_threshold=0.4)
        self.chunker.embed(self.texts)

    def test_cluster_output_structure(self):
        clusters = self.chunker.cluster()
        self.assertIsInstance(clusters, list)
        self.assertTrue(all(isinstance(c, list) for c in clusters))
        self.assertTrue(all(isinstance(item, str) for cluster in clusters for item in cluster))

    def test_vector_output(self):
        vector_output = self.chunker.get_vector_output()
        self.assertIsInstance(vector_output, list)
        self.assertTrue(all("text" in item and "embedding" in item and "id" in item and "cluster" in item for item in vector_output))
        self.assertTrue(all(isinstance(item["embedding"], list) for item in vector_output))

    def test_langchain_docs(self):
        docs = self.chunker.to_langchain_documents()
        from langchain.schema import Document
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))
        self.assertTrue(all("cluster" in doc.metadata for doc in docs))

if __name__ == '__main__':
    unittest.main()
