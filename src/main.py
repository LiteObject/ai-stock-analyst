"""Main entry point for the AI Stock Analyst trading system."""

# =============================================================================
# OpenTelemetry Tracing Setup (must be before other imports)
# =============================================================================
import os

# Enable LangSmith OpenTelemetry tracing
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

# =============================================================================

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402

import questionary  # noqa: E402
from colorama import Fore, Style, init  # noqa: E402
from dateutil.relativedelta import relativedelta  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from agents.fundamentals import fundamentals_agent  # noqa: E402
from agents.hedge_fund_manager import hedge_fund_manager_agent  # noqa: E402
from agents.portfolio_manager import portfolio_management_agent  # noqa: E402
from agents.risk_manager import risk_management_agent  # noqa: E402
from agents.sentiment import sentiment_agent  # noqa: E402
from agents.technicals import technical_analyst_agent  # noqa: E402
from agents.valuation import valuation_agent  # noqa: E402
from graph.state import AgentState  # noqa: E402
from utils.display import print_trading_output  # noqa: E402


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Style.DIM + Fore.WHITE,  # Gray/dim for INFO
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Add color based on log level
        color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL

        # Format the message
        message = super().format(record)
        return f"{color}{message}{reset}"


# Configure logging with colored output
init(autoreset=True)

handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
logger = logging.getLogger(__name__)


def parse_hedge_fund_response(response):
    """Parse the hedge fund response from JSON or extract from text."""
    if not response:
        return None

    # Try direct JSON parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    import re

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in the response
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Parse human-readable format as fallback
    result = {"action": "hold", "quantity": 0, "confidence": 0.5, "reasoning": response}

    # Extract action
    response_lower = response.lower()
    if "action:** buy" in response_lower or "action: buy" in response_lower:
        result["action"] = "buy"
    elif "action:** sell" in response_lower or "action: sell" in response_lower:
        result["action"] = "sell"
    elif "action:** hold" in response_lower or "action: hold" in response_lower:
        result["action"] = "hold"

    # Extract quantity
    qty_match = re.search(r"quantity[:\s*]+(\d[\d,]*)", response, re.IGNORECASE)
    if qty_match:
        result["quantity"] = int(qty_match.group(1).replace(",", ""))

    # Extract confidence
    conf_match = re.search(r"confidence[:\s*]+(\d+\.?\d*)", response, re.IGNORECASE)
    if conf_match:
        conf_value = float(conf_match.group(1))
        result["confidence"] = conf_value / 100 if conf_value > 1 else conf_value

    logger.info(f"Parsed response from text: action={result['action']}, quantity={result['quantity']}")
    return result


# === Run the Hedge Fund ===
def run_hedge_fund(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list = None,
    use_hedge_fund_manager: bool = False,
):
    """Execute the hedge fund trading workflow.

    Args:
        ticker: Stock ticker symbol
        start_date: Analysis start date (YYYY-MM-DD)
        end_date: Analysis end date (YYYY-MM-DD)
        portfolio: Current portfolio state
        show_reasoning: Whether to display agent reasoning
        selected_analysts: List of analysts to use (None = all)
        use_hedge_fund_manager: Use sophisticated hedge fund manager (default: False)

    Returns:
        Dict containing decision and analyst signals
    """
    logger.info(f"Running hedge fund analysis for {ticker} ({start_date} to {end_date})")

    # Create the workflow
    workflow = create_workflow(selected_analysts, use_hedge_fund_manager)
    agent = workflow.compile()

    try:
        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make a trading decision based on the provided data.",
                    )
                ],
                "data": {
                    "ticker": ticker,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                },
            },
        )

        logger.info(f"Hedge fund analysis completed for {ticker}")
        return {
            "decision": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    except Exception as e:
        logger.error(f"Error running hedge fund for {ticker}: {e}")
        raise


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None, use_hedge_fund_manager=False):
    """Create the workflow with selected analysts.

    Args:
        selected_analysts: List of analyst keys to include
        use_hedge_fund_manager: If True, use the sophisticated hedge fund manager
                               instead of the basic portfolio manager

    Returns:
        Compiled workflow
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = [
            "technical_analyst",
            "fundamentals_analyst",
            "sentiment_analyst",
            "valuation_analyst",
        ]

    # Dictionary of all available analysts
    analyst_nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analyst_agent),
        "fundamentals_analyst": ("fundamentals_agent", fundamentals_agent),
        "sentiment_analyst": ("sentiment_agent", sentiment_agent),
        "valuation_analyst": ("valuation_agent", valuation_agent),
    }

    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk management
    workflow.add_node("risk_management_agent", risk_management_agent)

    # Choose between hedge fund manager and basic portfolio manager
    if use_hedge_fund_manager:
        workflow.add_node("hedge_fund_manager_agent", hedge_fund_manager_agent)
        final_agent = "hedge_fund_manager_agent"
    else:
        workflow.add_node("portfolio_management_agent", portfolio_management_agent)
        final_agent = "portfolio_management_agent"

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", final_agent)
    workflow.add_edge(final_agent, END)

    workflow.set_entry_point("start_node")
    return workflow


# # Initialize app as None - it will be set in __main__
# app = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument(
        "--hedge-fund-manager",
        action="store_true",
        help="Use sophisticated hedge fund manager instead of basic portfolio manager",
    )

    args = parser.parse_args()

    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[
            questionary.Choice("Technical Analyst", value="technical_analyst"),
            questionary.Choice("Fundamentals Analyst", value="fundamentals_analyst"),
            questionary.Choice("Sentiment Analyst", value="sentiment_analyst"),
            questionary.Choice("Valuation Analyst", value="valuation_analyst"),
        ],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
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

    # Ask about hedge fund manager mode
    use_hedge_fund_manager = args.hedge_fund_manager
    if not use_hedge_fund_manager:
        use_hedge_fund_manager = questionary.confirm(
            "Use sophisticated Hedge Fund Manager? (includes news search & signal synthesis)",
            default=True,
            style=questionary.Style([("answer", "fg:green")]),
        ).ask()

    if use_hedge_fund_manager:
        print(f"\n{Fore.GREEN}✓ Using Hedge Fund Manager mode{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}→ Using basic Portfolio Manager mode{Style.RESET_ALL}")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts, use_hedge_fund_manager)
    app = workflow.compile()

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # TODO: Make this configurable via args
    portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0,  # No initial stock position
    }

    # Run the hedge fund
    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        use_hedge_fund_manager=use_hedge_fund_manager,
    )
    print_trading_output(result)
