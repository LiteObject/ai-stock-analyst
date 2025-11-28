"""
Reusable UI components for the AI Stock Analyst Streamlit app.
"""

import streamlit as st


def apply_custom_theme():
    """Apply custom CSS theme matching the original design."""
    st.markdown(
        """
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #00D4AA;
            --secondary-color: #1E1E1E;
            --background-color: #0E1117;
            --text-color: #FAFAFA;
            --success-color: #00D4AA;
            --warning-color: #FFD700;
            --error-color: #FF6B6B;
        }

        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #00D4AA 0%, #00A3CC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        /* Signal cards */
        .signal-card {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid;
        }

        .signal-bullish {
            border-left-color: #00D4AA;
            background-color: rgba(0, 212, 170, 0.1);
        }

        .signal-bearish {
            border-left-color: #FF6B6B;
            background-color: rgba(255, 107, 107, 0.1);
        }

        .signal-neutral {
            border-left-color: #FFD700;
            background-color: rgba(255, 215, 0, 0.1);
        }

        /* Metric cards */
        .metric-card {
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00D4AA;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #888;
        }

        /* Trading decision box */
        .decision-box {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        }

        .decision-buy {
            border: 2px solid #00D4AA;
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.3);
        }

        .decision-sell {
            border: 2px solid #FF6B6B;
            box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
        }

        .decision-hold {
            border: 2px solid #FFD700;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }

        /* Confidence meter */
        .confidence-bar {
            background-color: #2D2D2D;
            border-radius: 10px;
            height: 10px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        /* Agent cards */
        .agent-card {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .agent-name {
            font-weight: bold;
            color: #00D4AA;
            margin-bottom: 0.5rem;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Custom button styling */
        .stButton > button {
            background: linear-gradient(90deg, #00D4AA 0%, #00A3CC 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 170, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">🤖 AI Stock Analyst</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="color: #888; font-size: 1.1rem;">
        An AI-powered stock analysis tool with multiple analyst agents working together
        to make intelligent trading decisions.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Render the sidebar configuration and return settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Agent selection
        st.subheader("🤖 Analyst Agents")

        selected_analysts = []

        if st.checkbox(
            "Technical Analyst",
            value=True,
            help="Analyzes price patterns and technical indicators",
        ):
            selected_analysts.append("technical_analyst")

        if st.checkbox(
            "Fundamentals Analyst",
            value=True,
            help="Examines financial statements and metrics",
        ):
            selected_analysts.append("fundamentals_analyst")

        if st.checkbox("Sentiment Analyst", value=True, help="Analyzes market sentiment and news"):
            selected_analysts.append("sentiment_analyst")

        if st.checkbox("Valuation Analyst", value=True, help="Calculates intrinsic value"):
            selected_analysts.append("valuation_analyst")

        st.divider()

        # Advanced options
        st.subheader("🔧 Advanced Options")

        use_hedge_fund_manager = st.checkbox(
            "Use Hedge Fund Manager",
            value=True,
            help="Use sophisticated hedge fund manager with news integration and signal synthesis",
        )

        show_reasoning = st.checkbox(
            "Show Agent Reasoning",
            value=True,
            help="Display detailed reasoning from agents",
        )

        st.divider()

        # Info section
        st.subheader("ℹ️ About")
        st.markdown(
            """
            This AI Stock Analyst uses multiple specialized agents:

            - **Technical Analyst**: Charts & patterns
            - **Fundamentals Analyst**: Financial health
            - **Sentiment Analyst**: Market mood
            - **Valuation Analyst**: Fair value

            All signals are synthesized by the Portfolio/Hedge Fund Manager
            to make final trading decisions.
            """
        )

        # Free tickers notice
        st.info("💡 **Free Tickers**: AAPL, GOOGL, MSFT, NVDA, TSLA work without an API key.")

    return {
        "selected_analysts": selected_analysts if selected_analysts else None,
        "use_hedge_fund_manager": use_hedge_fund_manager,
        "show_reasoning": show_reasoning,
    }


def render_analyst_signals(analyst_signals: dict):
    """Render analyst signals as styled cards."""
    if not analyst_signals:
        st.info("No analyst signals available.")
        return

    for agent_name, signal_data in analyst_signals.items():
        # Determine signal type
        signal = signal_data.get("signal", "neutral").lower()
        confidence = signal_data.get("confidence", 0)

        # Set icon based on signal
        if signal == "bullish":
            icon = "🟢"
        elif signal == "bearish":
            icon = "🔴"
        else:
            icon = "🟡"

        # Format agent name
        display_name = agent_name.replace("_", " ").title()

        # Create expandable card
        with st.expander(
            f"{icon} {display_name} - {signal.upper()} ({confidence:.0%} confidence)",
            expanded=False,
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Signal", signal.upper())
                st.metric("Confidence", f"{confidence:.0%}")

            with col2:
                # Show reasoning if available
                reasoning = signal_data.get("reasoning", "No reasoning provided.")
                st.markdown("**Reasoning:**")
                st.markdown(f"_{reasoning}_")

            # Show additional data if available
            additional_data = {k: v for k, v in signal_data.items() if k not in ["signal", "confidence", "reasoning"]}

            if additional_data:
                st.markdown("**Additional Data:**")
                st.json(additional_data)


def render_trading_decision(decision: dict):
    """Render the trading decision with styling."""
    if not decision:
        st.info("No trading decision available.")
        return

    action = decision.get("action", "hold").lower()
    quantity = decision.get("quantity", 0)
    confidence = decision.get("confidence", 0)
    reasoning = decision.get("reasoning", "No reasoning provided.")

    # Determine styling based on action
    if action == "buy":
        color = "#00D4AA"
        icon = "📈"
        box_class = "decision-buy"
    elif action == "sell":
        color = "#FF6B6B"
        icon = "📉"
        box_class = "decision-sell"
    else:
        color = "#FFD700"
        icon = "⏸️"
        box_class = "decision-hold"

    # Decision box
    st.markdown(
        f"""
        <div class="decision-box {box_class}">
            <h1 style="color: {color}; font-size: 3rem;">{icon} {action.upper()}</h1>
            <p style="font-size: 1.5rem; color: #FAFAFA;">Quantity: <strong>{quantity}</strong> shares</p>
            <div style="margin: 1rem 0;">
                <p style="color: #888;">Confidence Level</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence * 100}%; background: {color};"></div>
                </div>
                <p style="color: {color}; font-size: 1.2rem; margin-top: 0.5rem;">{confidence:.0%}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Reasoning section
    st.subheader("📝 Decision Reasoning")
    st.markdown(reasoning)

    # Show full decision data
    with st.expander("📊 Full Decision Data"):
        st.json(decision)


def render_portfolio_chart(decision: dict):
    """Render portfolio visualization."""
    import matplotlib.pyplot as plt

    if not decision:
        st.info("No portfolio data available.")
        return

    action = decision.get("action", "hold").lower()
    quantity = decision.get("quantity", 0)

    # Create a simple visualization
    fig, ax = plt.subplots(figsize=(10, 4))

    # Simple bar chart showing the action
    actions = ["Buy", "Hold", "Sell"]
    values = [
        1 if action == "buy" else 0,
        1 if action == "hold" else 0,
        1 if action == "sell" else 0,
    ]
    bar_colors = ["#00D4AA", "#FFD700", "#FF6B6B"]

    bars = ax.bar(actions, values, color=bar_colors, alpha=0.3)

    # Highlight the selected action
    action_index = {"buy": 0, "hold": 1, "sell": 2}.get(action, 1)
    bars[action_index].set_alpha(1.0)

    ax.set_ylabel("Selected Action")
    ax.set_title(f"Trading Decision: {action.upper()} {quantity} shares")
    ax.set_ylim(0, 1.5)

    # Style the chart
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig)
    plt.close()

    # Portfolio metrics
    st.subheader("📊 Portfolio Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Action", action.upper(), delta=None)

    with col2:
        st.metric("Quantity", f"{quantity} shares", delta=None)

    with col3:
        confidence = decision.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.0%}", delta=None)
