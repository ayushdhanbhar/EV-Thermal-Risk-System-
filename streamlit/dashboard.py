import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


def render_dashboard(sequence, prediction):
    temperatures = sequence[:, 0]
    currents = sequence[:, 1]
    voltages = sequence[:, 2]
    rpms = sequence[:, 3]

    st.subheader("Battery Signals")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{temperatures[-1]:.2f} C")
    col2.metric("Current", f"{currents[-1]:.2f} A")
    col3.metric("Voltage", f"{voltages[-1]:.2f} V")
    col4.metric("Motor RPM", f"{rpms[-1]:.2f}")

    st.subheader("Thermal Risk")
    st.markdown(f"### {prediction['risk_level']}")
    st.metric("Risk Probability", f"{prediction['risk_probability'] * 100:.2f}%")
    st.info(f"Advisory: {prediction['advisory']}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(temperatures)
    axes[0, 0].set_title("Battery Temperature")
    axes[0, 1].plot(currents)
    axes[0, 1].set_title("Battery Current")
    axes[1, 0].plot(voltages)
    axes[1, 0].set_title("Battery Voltage")
    axes[1, 1].plot(rpms)
    axes[1, 1].set_title("Motor RPM")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Risk Probability Breakdown")
    probs_df = st.session_state.get("probs_df")
    if probs_df is not None:
        st.bar_chart(probs_df)
    else:
        st.write(prediction["probabilities"])
