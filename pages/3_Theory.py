import streamlit as st

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

st.markdown('''
# Electrocardiograms (ECGs) in Medicine
### What is an ECG?
An electrocardiogram (ECG) is a test that records the electrical activity of the heart over time using electrodes placed on the skin. The resulting ECG tracing depicts the heart rhythm and electrical conduction system.

Key components of an ECG include:

- P wave
- QRS complex
- T wave
- PR interval
- QT interval
These waveforms and intervals reflect the electrical impulses as they travel through the heart.

Current Uses of ECGs in Medicine
ECGs have many applications in healthcare:

- Evaluating heart rhythm
	- Can detect abnormal rhythms like atrial fibrillation, ventricular tachycardia, bradycardia
- Detecting myocardial ischemia or infarction
	- Changes in the ST segment can indicate a heart attack
- Measuring effects of cardiac drugs
	- ECGs show how medications impact heart rate, rhythm
- Checking pacemakers
	- Assess if pacemakers are functioning properly
- Cardiac stress testing
	- ECGs monitored during exercise stress tests
- Monitoring during surgery
	- Detect ischemia during procedures

### Interesting Facts about ECGs
German physiologist Einthoven invented the first practical ECG machine in 1903.

ECG paper moves at 25 mm/s, so each small block = 0.04 s.

Electrical signals between heart and skin only 0.0001 to 0.0004 volts.

Causes of artifacts on tracings:

Patient movement
Loose electrodes
Normal heart rates on ECG:

60-100 beats per minute in adults
In summary, the ECG is a fundamental tool used ubiquitously in cardiology to assess the heart's electrical activity, diagnose conditions, and monitor patients. It provides crucial data for managing heart disease.''')