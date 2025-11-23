import streamlit as st
import time

TIMEOUT = 30  # 30 seconds timeout

def password_page():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "last_interaction" not in st.session_state:
        st.session_state.last_interaction = time.time()

    # Check timeout
    if st.session_state.authenticated:
        if time.time() - st.session_state.last_interaction > TIMEOUT:
            st.session_state.authenticated = False
            st.rerun()  # Go back to password page
    
    # Show password input if not authenticated
    if not st.session_state.authenticated:
        pwd = st.text_input("Enter password:", type="password")
        #if pwd == st.secrets.get("APP_PASSWORD"):
        if pwd == st.secrets["passwords"]["admin"]:
            st.session_state.authenticated = True
            st.session_state.last_interaction = time.time()
            st.rerun()  # Go to main app
        elif pwd != "":
            st.warning("❌ Incorrect password. Try again.")
