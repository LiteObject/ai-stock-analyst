"""
AI Stock Analyst - Streamlit Web Interface

A sophisticated AI-powered stock analysis tool with multiple analyst agents.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components import (  # noqa: E402
    apply_custom_theme,
    render_analyst_signals,
    render_header,
    render_portfolio_chart,
    render_sidebar,
    render_trading_decision,
)

# Page configuration
st.set_page_config(
    page_title="AI Stock Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme
apply_custom_theme()


def run_analysis(ticker: str, start_date: str, end_date: str, portfolio: dict, config: dict) -> dict:
    """Run the hedge fund analysis."""
    from main import run_hedge_fund

    return run_hedge_fund(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=config.get("show_reasoning", True),
        selected_analysts=config.get("selected_analysts"),
        use_hedge_fund_manager=config.get("use_hedge_fund_manager", False),
    )


def main():
    """Main Streamlit application."""
    # Render header
    render_header()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Stock Analysis")

        # Ticker input
        ticker = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)",
        ).upper()

        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            start_date_input = st.date_input("Start Date", value=start_date, help="Analysis start date")
        with col_end:
            end_date_input = st.date_input("End Date", value=end_date, help="Analysis end date")

    with col2:
        st.subheader("💰 Portfolio")

        # Portfolio configuration
        initial_cash = st.number_input(
            "Initial Cash ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
        )

        initial_stock = st.number_input("Initial Stock Position", min_value=0, max_value=10000, value=0, step=10)

    # Horizontal divider
    st.divider()

    # Analysis button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not ticker:
            st.error("Please enter a stock ticker symbol.")
            return

        # Create portfolio
        portfolio = {
            "cash": float(initial_cash),
            "stock": int(initial_stock),
        }

        # Progress container
        progress_container = st.container()

        with progress_container:
            with st.spinner(f"Analyzing {ticker}..."):
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Update progress
                    status_text.text("🔄 Initializing agents...")
                    progress_bar.progress(10)

                    # Run analysis
                    status_text.text("📊 Running technical analysis...")
                    progress_bar.progress(25)

                    status_text.text("📈 Analyzing fundamentals...")
                    progress_bar.progress(40)

                    status_text.text("💭 Evaluating sentiment...")
                    progress_bar.progress(55)

                    status_text.text("💰 Calculating valuation...")
                    progress_bar.progress(70)

                    # Actually run the analysis
                    result = run_analysis(
                        ticker=ticker,
                        start_date=start_date_input.strftime("%Y-%m-%d"),
                        end_date=end_date_input.strftime("%Y-%m-%d"),
                        portfolio=portfolio,
                        config=config,
                    )

                    status_text.text("⚖️ Assessing risk...")
                    progress_bar.progress(85)

                    status_text.text("🎯 Making trading decision...")
                    progress_bar.progress(100)

                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()

                    # Store result in session state
                    st.session_state["last_result"] = result
                    st.session_state["last_ticker"] = ticker

                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
                    return

    # Display results if available
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        ticker = st.session_state.get("last_ticker", "")

        st.divider()

        # Results header
        st.header(f"📋 Analysis Results for {ticker}")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🎯 Trading Decision", "📊 Analyst Signals", "📈 Portfolio"])

        with tab1:
            render_trading_decision(result.get("decision", {}))

        with tab2:
            render_analyst_signals(result.get("analyst_signals", {}))

        with tab3:
            render_portfolio_chart(result.get("decision", {}))


if __name__ == "__main__":
    main()
