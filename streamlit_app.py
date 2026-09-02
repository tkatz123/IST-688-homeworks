import streamlit as st 

hw_1 = st.Page(
    'HW/HW1.py',
    title = 'HW 1'
)

hw_2 = st.Page(
    'HW/HW2.py',
    title = 'HW 2',
    default = True
)

pg = st.navigation([hw_1, hw_2])
st.set_page_config(page_title= 'HW Manager', page_icon=None, layout='centered', initial_sidebar_state='expanded')
pg.run()