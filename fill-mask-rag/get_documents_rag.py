from typing import List
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: str, db):
    question = "What are the best values for temperature, humidity and soil ph for {input1}?"
    searchDocs = db.similarity_search(question)
    return searchDocs[0].page_content
