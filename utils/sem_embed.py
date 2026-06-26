from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def extract_semantic_embeddings(items, model_path):
    """
    Calculate the embedding for a given word/sentece list.

    Parameters
    ----------
    items : list of str
        The input words/sentences.
    model_path : str|Path
        The path to the model for semantic embedding.
    
    Returns
    -------
    embeddings : np.float32
        Semantic embeddings for the given words/sentences. Shape: (n_items, n_dims)

    """
    model = SentenceTransformer(model_name_or_path=model_path, local_files_only=True)
    model.eval()

    embeddings = model.encode(items, normalize_embeddings=True)
    
    return embeddings


if __name__ == "__main__":

    # Word Test
    test1, test2, test3, test4 = "优秀", "卓越", "糟糕", "湿润"
    print(f"\nTest: {test1}|{test2}|{test3}|{test4}\n")
    items = [test1, test2, test3, test4]

    ## bge-large-zh-v1.5
    model_name = "./huggingface/sentence-transformers/bge-large-zh-v1.5"
    embeds = extract_semantic_embeddings(items, model_name)
    print("\nSimilarity Results for Model: bge-large-zh-v1.5")
    sim_matrix = cosine_similarity(embeds)
    print(sim_matrix)

    ## paraphrase-multilingual-MiniLM-L12-v2
    model_name = "./huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeds = extract_semantic_embeddings(items, model_name)
    print("\nSimilarity Results for Model: paraphrase-multilingual-MiniLM-L12-v2")
    sim_matrix = cosine_similarity(embeds)
    print(sim_matrix)

    # Sentence Test
    test1, test2, test3, test4 = "因为他努力学习", "虽然他努力学习", "因为不努力学习", "因为加班了很久"
    print(f"\nTest: {test1}|{test2}|{test3}|{test4}\n")
    items = [test1, test2, test3, test4]

    ## bge-large-zh-v1.5
    model_name = "./huggingface/sentence-transformers/bge-large-zh-v1.5"
    embeds = extract_semantic_embeddings(items, model_name)
    print("\nSimilarity Results for Model: bge-large-zh-v1.5")
    sim_matrix = cosine_similarity(embeds)
    print(sim_matrix)

    ## paraphrase-multilingual-MiniLM-L12-v2
    model_name = "./huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeds = extract_semantic_embeddings(items, model_name)
    print("\nSimilarity Results for Model: paraphrase-multilingual-MiniLM-L12-v2")
    sim_matrix = cosine_similarity(embeds)
    print(sim_matrix)