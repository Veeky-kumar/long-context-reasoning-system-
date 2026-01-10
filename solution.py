"""
KDSH 2026 Track A Submission
System: Pathway-Enhanced Narrative Reasoner
Status: Track A Compliant (Pathway ETL + Semantic Search)
Optimization: Robust Pathway Temp Handling + Balanced Prompting
"""

import pandas as pd
import json
import os
from pathlib import Path
import logging
import re
from tqdm import tqdm
import requests
import numpy as np
import tempfile
import shutil
import time

# Semantic Search
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# Pathway
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CHUNKING
# -----------------------------------------------------------------------------
def chunk_text_udf(text, metadata, chunk_size=300, overlap=50):
    if not isinstance(text, str) or not text: return []
    
    path = "unknown"
    if hasattr(metadata, "path"): path = metadata.path
    elif isinstance(metadata, dict): path = metadata.get("path", "unknown")
        
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append({
            'text': chunk_text, 
            'metadata': {'source': str(path), 'start': i}
        })
    return chunks

# -----------------------------------------------------------------------------
# DOCUMENT STORE
# -----------------------------------------------------------------------------
class PathwayDocumentStore:
    def __init__(self):
        self.embedder = None
        if SEMANTIC_AVAILABLE:
            logger.info("⚡ Initializing Embedding Model (all-mpnet-base-v2)...")
            self.embedder = SentenceTransformer('all-mpnet-base-v2')

    def ingest_narrative(self, narrative_path: str):
        chunks = self._get_chunks(narrative_path)
        
        if self.embedder and chunks:
            logger.info(f"   ↳ Generating embeddings for {len(chunks)} chunks...")
            texts = [c['text'] for c in chunks]
            embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=True)
            for i, chunk in enumerate(chunks):
                chunk['vector'] = embeddings[i]
        return chunks

    def _get_chunks(self, narrative_path):
        if PATHWAY_AVAILABLE:
            return self._pathway_chunking(narrative_path)
        return self._fallback_chunking(narrative_path)

    def _pathway_chunking(self, narrative_path):
        """
        Track A Compliant Ingestion with Robust Temp File Handling
        """
        temp_dir = None
        try:
            parent_dir = os.path.dirname(narrative_path) or "."
            fname = os.path.basename(narrative_path)
            
            # 1. Read Binary (Whole File)
            documents = pw.io.fs.read(
                parent_dir, glob=fname, format="binary", mode="static", with_metadata=True
            )
            
            # 2. Decode & Chunk
            def process_file(data, meta):
                try:
                    text = data.decode('utf-8', errors='ignore')
                    return chunk_text_udf(text, meta)
                except: return []

            chunks_table = documents.select(
                res=pw.apply(process_file, pw.this.data, pw.this._metadata)
            ).flatten(pw.this.res)
            
            # 3. Materialize safely using mkdtemp
            # This creates a unique directory every time to prevent race conditions
            os.makedirs("/app/output", exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix="pw_job_", dir="/app/output")
            out_path = os.path.join(temp_dir, "out.csv")
            
            # Write and Execute
            pw.io.csv.write(chunks_table, out_path)
            pw.run()
            
            # Read back results
            clean_chunks = []
            if os.path.isdir(out_path):
                # Pathway created a directory of CSVs
                csv_files = list(Path(out_path).glob("*.csv"))
                if csv_files:
                    df = pd.concat([pd.read_csv(f) for f in csv_files])
                else:
                    df = pd.DataFrame()
            elif os.path.exists(out_path):
                # Pathway created a single file
                df = pd.read_csv(out_path)
            else:
                df = pd.DataFrame()

            # Parse results
            if not df.empty:
                col0 = df.columns[0]
                for _, row in df.iterrows():
                    val = row[col0]
                    if isinstance(val, str):
                        try: val = eval(val)
                        except: continue
                    if isinstance(val, dict) and 'text' in val:
                        clean_chunks.append(val)
            
            if clean_chunks:
                logger.info(f"✓ Pathway ingested {len(clean_chunks)} chunks")
                return clean_chunks
            else:
                return self._fallback_chunking(narrative_path)

        except Exception as e:
            logger.warning(f"Pathway error: {e}")
            return self._fallback_chunking(narrative_path)
        finally:
            # Clean up the unique temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _fallback_chunking(self, path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return chunk_text_udf(f.read(), {'path': path})
        except: return []

# -----------------------------------------------------------------------------
# EVALUATOR
# -----------------------------------------------------------------------------
class ConsistencyEvaluator:
    def __init__(self, model_name):
        self.store = PathwayDocumentStore()
        self.llm_model = model_name
        self.indices = {}
        self.api_url = "http://host.docker.internal:11434/api/generate"
        if not os.path.exists('/.dockerenv'):
            self.api_url = "http://localhost:11434/api/generate"

    def load_data(self, novels_dir):
        for f in Path(novels_dir).glob("*.txt"):
            key = f.stem.lower().replace("_", " ")
            logger.info(f"📖 Indexing: {f.name}")
            self.indices[key] = self.store.ingest_narrative(str(f))

    def get_llm_response(self, prompt):
        try:
            res = requests.post(self.api_url, json={
                "model": self.llm_model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.1, "num_predict": 400}
            }, timeout=120)
            if res.status_code == 200:
                return res.json().get('response', '')
        except: pass
        return ""

    def evaluate(self, row):
        book_key = next((k for k in self.indices if k in row['book_name'].lower().replace("_", " ")), None)
        chunks = self.indices.get(book_key, [])
        context = ""
        
        if chunks and self.store.embedder:
            q_vec = self.store.embedder.encode([row['content']])
            c_vecs = np.array([c['vector'] for c in chunks])
            sims = cosine_similarity(q_vec, c_vecs)[0]
            top_idxs = sims.argsort()[-10:][::-1]
            context = "\n\n".join([chunks[i]['text'] for i in top_idxs])
        
        # BALANCED PROMPT
        prompt = f"""Task: Check consistency between Backstory and Novel.

NOVEL EXCERPTS:
{context[:12000]}

BACKSTORY CLAIM:
{row['content']}

INSTRUCTIONS:
1. Determine if the Backstory directly CONTRADICTS the Novel.
2. A contradiction implies a logical impossibility (e.g. wrong dates, dead character is alive, wrong relationship).
3. If the backstory adds new details that fit plausibly, it is CONSISTENT.
4. If the novel excerpts do not mention the event, assume CONSISTENT.

DECISION:
- Return 0 (Contradict) ONLY if there is clear opposing evidence.
- Return 1 (Consistent) if plausible or not mentioned.

JSON OUTPUT: {{"prediction": 0 or 1, "reasoning": "brief explanation"}}"""
        
        resp = self.get_llm_response(prompt)
        pred = 1
        
        if '"prediction": 0' in resp or '"prediction":0' in resp: pred = 0
        
        return {'Story ID': row['id'], 'Prediction': pred, 'Rationale': resp[:200]}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='test.csv')
    parser.add_argument('--novels', default='novels/')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--model', default='qwen2.5:7b')
    args = parser.parse_args()
    
    evaluator = ConsistencyEvaluator(model_name=args.model)
    evaluator.load_data(args.novels)
    
    df = pd.read_csv(args.test)
    results = [evaluator.evaluate(row) for _, row in tqdm(df.iterrows(), total=len(df))]
    
    pd.DataFrame(results).to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()