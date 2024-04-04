from typing import List
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(list_of_docs: str, crop: str) -> str:
    prompt = list_of_docs

    prompt += "\\Fill in with the best numerical values for temperature, humidity and soil_ph to grow " + crop + " {{'crop' : " + crop + " , 'temperature' : [MASK], 'humidity' : [MASK], 'soil_pH' : [MASK]}}"

    return prompt