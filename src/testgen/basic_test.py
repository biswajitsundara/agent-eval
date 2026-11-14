from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Generate goldens
synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["faq.txt"],
    include_expected_output=True,
    max_goldens_per_context=2,
)

# Optional: save to disk for reuse
synthesizer.save_as(
    file_type="json",
    directory="./synthetic_data",
    file_name="synthetic_data",
)

# 2. Wrap in a dataset (no reload needed)
dataset = EvaluationDataset(goldens=goldens)


# 3. LangChain chain: prompt -> model -> string
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful shoe store support assistant. Be concise."),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | StrOutputParser()

# 4. Turn goldens into test cases
test_cases = []
for golden in dataset.goldens:
    test_cases.append(
        LLMTestCase(
            input=golden.input,
            actual_output=chain.invoke({"question": golden.input}),
            expected_output=golden.expected_output,
        )
    )

# 4b. Peek at what you're evaluating
for tc in test_cases[:3]:
    print("Q:", tc.input)
    print("A:", tc.actual_output)
    print("-" * 40)

# 5. Evaluate with one metric
metric = AnswerRelevancyMetric(threshold=0.7, include_reason=True)
evaluate(test_cases, [metric])