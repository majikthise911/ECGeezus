import streamlit as st
from streamlit_option_menu import option_menu # pip install streamlit-option-menu


# from dotenv import load_dotenv # comment out for deployment 
# load_dotenv() # comment out for deployment 

import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px	# pip install plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns	# pip install seaborn
from datetime import datetime, timedelta
from io import BytesIO # for downloading files

# -------------- PAGE CONFIG --------------
page_title = "ECGeezus"
page_icon = ":zap:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title = page_title, layout = layout, page_icon = page_icon)
st.title(page_title + " " + page_icon)


######################################################8/1/23############################################################################
st.markdown("## Analysis (can take up to 2 min to load)")
from dotenv import load_dotenv
load_dotenv()

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Update imports
from openai.error import InvalidRequestError

# New function to get portfolio analysis  
def get_portfolio_analysis():

  # Craft prompt with portfolio weights
  prompt = f'''You are a cardiologist writing an analysis report for a patient's electrocardiogram (ECG) results. Please write a detailed analysis report in Markdown formatting that includes the following sections:

# Summary
- One paragraph summarizing the overall findings, heart rate and rhythm, intervals, abnormalities etc.

# Detailed Analysis 
- Detailed analysis of each waveform segment (P wave, PR interval, QRS complex, ST segment, T wave etc.)
- Any abnormalities or irregularities observed
- Comparison to normal ECG ranges
- Interpretation of findings (e.g. signs of ischemia, infarction, arrhythmia etc.)

# Recommendations
- Suggested next steps based on your analysis (e.g. follow up testing, treatment options etc.)

# Example ECG
- Provide a sample 12-lead ECG image with annotated waveform segments 

Please write the report in a professional tone using medical terminology where appropriate. The report should demonstrate your expertise as a cardiologist analyzing ECG results for a patient.'''
  
  try:
    # Use Chat Completions endpoint
    response = openai.ChatCompletion.create(
      model="gpt-4",  
      messages=[
        {"role": "system", "content": "You are an investment advisor giving portfolio analysis"},
        {"role": "user", "content": prompt}
      ]
    )
  except InvalidRequestError as e:
    # Handle incorrect API endpoint error
    print(e)
    return None
  
  return response.choices[0].message.content
ecg_analysis = get_portfolio_analysis()


with st.expander("Portfolio Analysis"):
    st.markdown(ecg_analysis)

######################################################8/1/23############################################################################


st.markdown("""---""")
st.markdown('Made with :heart: by [Jordan Clayton](https://dca-rsi.streamlit.app/Contact_Me)')
st.markdown("""---""")
