multiagent-rag/
├── agents/
│   ├── segmenter.py
│   ├── image_relevance.py
│   ├── image_handler.py
│   ├── retriever.py
│   ├── verifier.py
│   └── generator.py
├── data/
│   └── Round1_Task.pdf
├── tests/
│   └── test_pipeline.py
├── main.py
├── config.yaml
├── requirements.txt
├── evaluation.py
├── README.md
└── docs/
    └── Technical_Architecture_Report.pdf

#agents/segmenter.py


import fitz  # PyMuPDF

class DocumentSegmenter:
    def segment(self, document_path):
        doc = fitz.open(document_path)
        segments = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            images = page.get_images(full=True)
            segments.append({
                "page": page_num + 1,
                "text": text,
                "images": images
            })
        return segments
        
#agents/image_relevance.py
class ImageRelevanceAgent:
    def classify(self, segments, query):
        for seg in segments:
            seg['image_relevance'] = []
            for img in seg['images']:
                if any(word in query.lower() for word in ['chart', 'figure', 'diagram', 'image']):
                    seg['image_relevance'].append('high')
                else:
                    seg['image_relevance'].append('low')
        return segments
#agents/image_handler.py

class ImageHandler:
    def process(self, segment):
        if 'high' in segment['image_relevance']:
            return None, "Image unprocessable, deferred to human review."
        return None, "No relevant image to process."
#agents/retriever.py
from sentence_transformers import SentenceTransformer, util

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def retrieve(self, segments, query):
        query_emb = self.model.encode(query)
        best_score = -1
        best_segment = None
        for seg in segments:
            text_emb = self.model.encode(seg['text'])
            score = util.pytorch_cos_sim(query_emb, text_emb).item()
            if score > best_score:
                best_score = score
                best_segment = seg
        return best_segment
#agents/verifier.py
class Verifier:
    def check(self, segment, query):
        if not segment or (segment['text'] == "" and 'high' in segment.get('image_relevance', [])):
            return False, "Insufficient information."
        return True, "Information sufficient."
#agents/generator.py
class ResponseGenerator:
    def generate(self, segment, query, verification_result):
        if not verification_result[0]:
            return {
                "answer": "Unable to answer conclusively: critical image content is not accessible for parsing.",
                "confidence": 0.0,
                "uncertainty_flag": True,
                "notes": verification_result[1]
            }
        answer = segment['text'][:200]
        return {
            "answer": answer,
            "confidence": 0.9,
            "uncertainty_flag": False,
            "notes": "Answer generated from text context."
        }
#main.py
from agents.segmenter import DocumentSegmenter
from agents.image_relevance import ImageRelevanceAgent
from agents.image_handler import ImageHandler
from agents.retriever import Retriever
from agents.verifier import Verifier
from agents.generator import ResponseGenerator

def process_query(document_path, user_query):
    segments = DocumentSegmenter().segment(document_path)
    segments = ImageRelevanceAgent().classify(segments, user_query)
    for seg in segments:
        img_text, img_note = ImageHandler().process(seg)
        seg['img_text'] = img_text
        seg['img_note'] = img_note
    best_segment = Retriever().retrieve(segments, user_query)
    verification_result = Verifier().check(best_segment, user_query)
    answer = ResponseGenerator().generate(best_segment, user_query, verification_result)
    return answer

if __name__ == "__main__":
    import sys
    doc_path = sys.argv[1]
    query = sys.argv[2]
    print(process_query(doc_path, query))

