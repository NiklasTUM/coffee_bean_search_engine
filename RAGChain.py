from LLMInference import LLMInference
from Retriever import Retriever
import json

from constants import constants


class RAGChain:
    def __init__(self):
        self.retriever = Retriever()
        self.answer_generator = LLMInference()
        self.system_prompt = self.load_system_prompt()

    def load_system_prompt(self):
        """
        Loads the system prompt from the specified JSON file.

        Returns:
            str: The system prompt loaded from the JSON file.
        """
        try:
            with open(constants.prompt_template_path, 'r', encoding='utf-8') as file:
                prompt_data = json.load(file)
                self.system_prompt = prompt_data.get("system")
                if not self.system_prompt:
                    raise ValueError("System prompt not found in the JSON file.")
            print("System prompt loaded successfully.")
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            raise

        return self.system_prompt

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_prompt(self, context: str, question: str) -> list[dict]:
        prompt = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"Context:  {context}"
                                        f"Question: {question}"}
        ]

        return prompt

    def chain(self):
        question = "What are unicode scripts?"
        retrieved_context = self.retriever.retriever_from_llm.invoke(question)
        context = self.format_docs(retrieved_context)
        print(context)
        prompt = self.create_prompt(context, question)
        answer = self.answer_generator.inference(prompt)
        print(answer)


if __name__ == '__main__':
    rag_chain = RAGChain()
    rag_chain.chain()
