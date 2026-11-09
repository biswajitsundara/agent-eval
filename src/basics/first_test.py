from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
from deepeval.evaluate import CacheConfig


def test_answer_relevancy_metric():
    testcase = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris."
    )
    metric = AnswerRelevancyMetric(model= "gpt-4o-mini", threshold=0.7, include_reason=True)
    assert_test(testcase, metric)

    #evaluate(test_cases=[testcase], metrics=[metric], cache_config=CacheConfig(write_cache=False))


if __name__ == "__main__":
    test_answer_relevancy_metric()