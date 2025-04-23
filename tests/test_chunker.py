import unittest
from chonkychunker import TextChunker
from langchain.schema import Document

class TestTextChunker(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "The milk is spoiled.",
            "Eggs are boiled and tasty.",
            "Physics involves matter and energy.",
            "Salt is added for flavor.",
            "Thermonuclear reactions are powerful.",
            "Car is washed using detergent.",
            "Fast food includes fries and burgers.",
            "Acid reacts with base to form salt."
        ]
        self.chunker = TextChunker(metric='cosine', top_k=4, distance_threshold=0.5, max_tokens=50)
        self.chunker.embed(self.texts)

    def test_cluster_structure(self):
        clusters = self.chunker.cluster()
        self.assertIsInstance(clusters, list)
        for cluster in clusters:
            self.assertTrue(all(isinstance(text, str) for text in cluster))

    def test_vector_output_unmerged(self):
        output = self.chunker.get_vector_output(merge=False)
        self.assertIsInstance(output, list)
        self.assertTrue(all("text" in item and "embedding" in item and isinstance(item["embedding"], list) for item in output))

    def test_vector_output_merged(self):
        output = self.chunker.get_vector_output(merge=True)
        self.assertIsInstance(output, list)
        for item in output:
            self.assertIn("text", item)
            self.assertIn("embedding", item)
            self.assertIsInstance(item["embedding"], list)

    def test_langchain_documents_unmerged(self):
        docs = self.chunker.to_langchain_documents(merge=False)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))

    def test_langchain_documents_merged(self):
        docs = self.chunker.to_langchain_documents(merge=True)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs))
        self.assertTrue(all("cluster" in doc.metadata for doc in docs))

if __name__ == '__main__':
    unittest.main()
