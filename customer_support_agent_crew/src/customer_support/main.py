#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

from customer_support.crew import customer_support

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the research crew.
    """
    inputs = {
        "customer": "ABC Tech",
        "person": "Praveen Nair",
        "inquiry": "I need help with understanding my dedeuctible for the current year. "
                "Can you provide guidance?"
    }

    # Create and run the crew
    result = customer_support().crew().kickoff(inputs=inputs)

    # Print the result
    print("\n\n=== FINAL DECISION ===\n\n")
    print(result.raw)


if __name__ == "__main__":
    run()