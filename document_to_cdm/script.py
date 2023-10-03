# -*- coding: utf-8 -*-
"""
@author: Shasa Foster
https://www.linkedin.com/in/shasafoster
"""

if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())


fp = 'term_sheet_sample.txt'


# https://learn.microsoft.com/en-GB/azure/ai-services/openai/concepts/models
# https://learn.microsoft.com/en-GB/azure/ai-services/openai/overview
# https://learn.microsoft.com/en-gb/training/modules/explore-azure-openai/?wt.mc_id=acom_openaiintroduction_webpage_gdc

with open(fp,'r',encoding='utf-8') as file:
    file_data = file.read()