from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import pdfplumber


# Load the pre-trained language model
config = {'max_new_tokens': 100, 'temperature': 0}
llm = CTransformers(model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF', model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config)

# Define the prompt template
template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
Answer the question below from context below :
{context}
{question} [/INST] </s>"""


# Input question
question = "what is sun"
context = """
The Sun is the star at the center of the Solar System. It is a massive, hot ball of plasma, inflated and heated by
energy produced by nuclear fusion reactions at its core.
"""

# Define the prompt with the provided question and PDF content chunk
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Generate response based on the chunk and question
chunk_response = llm_chain.run({"question": question, "context": context})
print(chunk_response)
