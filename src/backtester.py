"""Backtesting module for the AI Stock Analyst."""

# =============================================================================
# OpenTelemetry Tracing Setup (must be before other imports)
# =============================================================================
import os

# Enable LangSmith OpenTelemetry tracing
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

# =============================================================================

import logging  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import questionary  # noqa: E402
from colorama import Fore, Style, init  # noqa: E402
from dateutil.relativedelta import relativedelta  # noqa: E402

from main import run_hedge_fund  # noqa: E402
from tools.api import get_price_data  # noqa: E402
from utils.display import format_backtest_row, print_backtest_results  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent,
        ticker,
        start_date,
        end_date,
        initial_capital,
        selected_analysts=None,
    ):
        self.agent = agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.selected_analysts = selected_analysts
        self.portfolio = {"cash": initial_capital, "stock": 0}
        self.portfolio_values = []

    def parse_agent_response(self, agent_output):
        """Parse agent output to extract trading decision."""
        try:
            import json

            decision = json.loads(agent_output)
            return decision
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing action: {agent_output[:100] if agent_output else 'None'}... Error: {e}")
            return "hold", 0

    def execute_trade(self, action, quantity, current_price):
        """Validate and execute trades based on portfolio constraints.

        Args:
            action: Trade action ('buy', 'sell', or 'hold')
            quantity: Number of shares to trade
            current_price: Current stock price

        Returns:
            Number of shares actually traded
        """
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                logger.debug(f"Executed buy: {quantity} shares at ${current_price:.2f}")
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    logger.debug(f"Executed partial buy: {max_quantity} shares (requested {quantity})")
                    return max_quantity
                logger.warning(f"Cannot execute buy: insufficient funds for {quantity} shares")
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.portfolio["stock"])
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                logger.debug(f"Executed sell: {quantity} shares at ${current_price:.2f}")
                return quantity
            logger.warning("Cannot execute sell: no shares to sell")
            return 0
        return 0

    def run_backtest(self):
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []

        print("\nStarting backtest...")

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            output = self.agent(
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
                selected_analysts=self.selected_analysts,
            )

            agent_decision = output["decision"]
            action, quantity = agent_decision["action"], agent_decision["quantity"]
            df = get_price_data(self.ticker, lookback_start, current_date_str)
            current_price = df.iloc[-1]["close"]

            # Execute the trade with validation
            executed_quantity = self.execute_trade(action, quantity, current_price)

            # Update total portfolio value
            total_value = self.portfolio["cash"] + self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value

            # Count signals from selected analysts only
            analyst_signals = output["analyst_signals"]

            # Count signals
            bullish_count = len([s for s in analyst_signals.values() if s.get("signal", "").lower() == "bullish"])
            bearish_count = len([s for s in analyst_signals.values() if s.get("signal", "").lower() == "bearish"])
            neutral_count = len([s for s in analyst_signals.values() if s.get("signal", "").lower() == "neutral"])

            print(f"Signal counts - Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}")

            # Format and add row
            table_rows.append(
                format_backtest_row(
                    date=current_date.strftime("%Y-%m-%d"),
                    ticker=self.ticker,
                    action=action,
                    quantity=executed_quantity,
                    price=current_price,
                    cash=self.portfolio["cash"],
                    stock=self.portfolio["stock"],
                    total_value=total_value,
                    bullish_count=bullish_count,
                    bearish_count=bearish_count,
                    neutral_count=neutral_count,
                )
            )

            # Display the updated table
            print_backtest_results(table_rows)

            # Record the portfolio value
            self.portfolio_values.append({"Date": current_date, "Portfolio Value": total_value})

    def analyze_performance(self):
        """
        Analyze backtest performance with comprehensive metrics.

        Calculates:
        - Total Return
        - Annualized Return
        - Sharpe Ratio
        - Sortino Ratio
        - Maximum Drawdown
        - Calmar Ratio
        - Win Rate
        - Profit Factor
        - Volatility

        Returns:
            DataFrame with portfolio values and daily returns
        """
        import numpy as np

        # Convert portfolio values to DataFrame
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # Calculate total return
        total_return = (self.portfolio["portfolio_value"] - self.initial_capital) / self.initial_capital

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change()
        daily_returns = performance_df["Daily Return"].dropna()

        # Trading days per year
        trading_days = 252

        # Calculate annualized return
        num_days = len(performance_df)
        annualized_return = ((1 + total_return) ** (trading_days / num_days)) - 1 if num_days > 0 else 0

        # Calculate volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(trading_days)

        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(trading_days) if std_daily_return > 0 else 0

        # Calculate Sortino Ratio (only considers downside volatility)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = (mean_daily_return / downside_std) * np.sqrt(trading_days) if downside_std > 0 else 0

        # Calculate Maximum Drawdown
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = performance_df["Portfolio Value"] / rolling_max - 1
        max_drawdown = drawdown.min()

        # Calculate Calmar Ratio (annualized return / max drawdown)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

        # Calculate Win Rate
        winning_days = len(daily_returns[daily_returns > 0])
        total_trading_days = len(daily_returns)
        win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0

        # Calculate Profit Factor (gross profits / gross losses)
        gross_profits = daily_returns[daily_returns > 0].sum()
        gross_losses = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

        # Calculate average win/loss
        avg_win = daily_returns[daily_returns > 0].mean() * 100 if winning_days > 0 else 0
        avg_loss = daily_returns[daily_returns < 0].mean() * 100 if len(negative_returns) > 0 else 0

        # Print performance summary
        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE SUMMARY".center(60))
        print("=" * 60)

        print(f"\n{'RETURNS':^60}")
        print("-" * 60)
        print(f"  Total Return:           {total_return * 100:>10.2f}%")
        print(f"  Annualized Return:      {annualized_return * 100:>10.2f}%")
        print(f"  Volatility (Annual):    {volatility * 100:>10.2f}%")

        print(f"\n{'RISK-ADJUSTED METRICS':^60}")
        print("-" * 60)
        print(f"  Sharpe Ratio:           {sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:          {sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:           {calmar_ratio:>10.2f}")
        print(f"  Maximum Drawdown:       {max_drawdown * 100:>10.2f}%")

        print(f"\n{'TRADING STATISTICS':^60}")
        print("-" * 60)
        print(f"  Win Rate:               {win_rate:>10.2f}%")
        print(f"  Profit Factor:          {profit_factor:>10.2f}")
        print(f"  Average Win:            {avg_win:>10.2f}%")
        print(f"  Average Loss:           {avg_loss:>10.2f}%")
        print(f"  Trading Days:           {total_trading_days:>10d}")

        print(f"\n{'PORTFOLIO':^60}")
        print("-" * 60)
        print(f"  Initial Capital:        ${self.initial_capital:>12,.2f}")
        print(f"  Final Value:            ${self.portfolio['portfolio_value']:>12,.2f}")
        print(f"  Profit/Loss:            ${self.portfolio['portfolio_value'] - self.initial_capital:>12,.2f}")

        print("=" * 60)

        # Plot the portfolio value over time
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Portfolio Value
        axes[0, 0].plot(
            performance_df.index,
            performance_df["Portfolio Value"],
            color="blue",
            linewidth=1.5,
        )
        axes[0, 0].set_title("Portfolio Value Over Time")
        axes[0, 0].set_ylabel("Portfolio Value ($)")
        axes[0, 0].grid(True, alpha=0.3)

        # Drawdown
        axes[0, 1].fill_between(drawdown.index, drawdown * 100, 0, color="red", alpha=0.3)
        axes[0, 1].plot(drawdown.index, drawdown * 100, color="red", linewidth=1)
        axes[0, 1].set_title("Drawdown Over Time")
        axes[0, 1].set_ylabel("Drawdown (%)")
        axes[0, 1].grid(True, alpha=0.3)

        # Daily Returns Distribution
        axes[1, 0].hist(
            daily_returns * 100,
            bins=30,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1, 0].axvline(x=0, color="red", linestyle="--", linewidth=1)
        axes[1, 0].set_title("Daily Returns Distribution")
        axes[1, 0].set_xlabel("Daily Return (%)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative Returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        axes[1, 1].plot(
            cumulative_returns.index,
            cumulative_returns * 100,
            color="green",
            linewidth=1.5,
        )
        axes[1, 1].set_title("Cumulative Returns")
        axes[1, 1].set_ylabel("Cumulative Return (%)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return performance_df


# === Run the Backtest ===
if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )

    args = parser.parse_args()

    selected_analysts = None
    choices = questionary.checkbox(
        "Use the Space bar to select/unselect analysts.",
        choices=[
            questionary.Choice("Technical Analyst", value="technical_analyst"),
            questionary.Choice("Fundamentals Analyst", value="fundamentals_analyst"),
            questionary.Choice("Sentiment Analyst", value="sentiment_analyst"),
            questionary.Choice("Valuation Analyst", value="valuation_analyst"),
        ],
        instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("You must select at least one analyst. Using all analysts by default.")
        selected_analysts = None
    else:
        selected_analysts = choices
        print(
            f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}"
        )

    # Create an instance of Backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        selected_analysts=selected_analysts,
    )

    # Run the backtesting process
    backtester.run_backtest()
    performance_df = backtester.analyze_performance()
