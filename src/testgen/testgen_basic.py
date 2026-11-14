from deepeval.synthesizer import Synthesizer
from dotenv import load_dotenv
load_dotenv()  

synthesizer = Synthesizer()

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["faq.txt"]
)

synthesizer.save_as(file_type="json", directory="./synthetic_data")