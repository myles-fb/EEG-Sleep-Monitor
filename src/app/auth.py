"""Authentication UI helpers for Streamlit pages."""

import streamlit as st

from services import auth_service


def require_auth() -> str:
    """Gate that shows login/signup if unauthenticated.

    Returns the current user_id. Calls st.stop() if not logged in.
    """
    if "user_id" in st.session_state and st.session_state["user_id"]:
        return st.session_state["user_id"]

    st.title("EEG Sleep Monitor")
    st.caption("Sign in to continue")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Email and password are required.")
                else:
                    user = auth_service.authenticate(email.strip(), password)
                    if user:
                        st.session_state["user_id"] = user.id
                        st.session_state["user_email"] = user.email
                        st.session_state["user_display_name"] = user.display_name
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_name = st.text_input("Display Name")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_pw")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_pw2")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if not new_name or not new_email or not new_password:
                    st.error("All fields are required.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        user = auth_service.create_user(
                            email=new_email.strip(),
                            password=new_password,
                            display_name=new_name.strip(),
                        )
                        st.session_state["user_id"] = user.id
                        st.session_state["user_email"] = user.email
                        st.session_state["user_display_name"] = user.display_name
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    st.stop()


def show_user_sidebar():
    """Show logged-in user info and logout button in sidebar."""
    display_name = st.session_state.get("user_display_name", "")
    email = st.session_state.get("user_email", "")

    st.sidebar.markdown(f"**{display_name}**")
    st.sidebar.caption(email)
    if st.sidebar.button("Logout"):
        for key in ("user_id", "user_email", "user_display_name"):
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.divider()
