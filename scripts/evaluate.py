#!/usr/bin/env python3
"""
Run full evaluation pipeline: fixed vs heading vs baseline.
Outputs CSV/JSON with retrieval and generation metrics.
"""

import asyncio

from hybrid_rag.config import settings
from hybrid_rag.evaluation.runner import run_evaluation

async def main():
    await run_evaluation(
        questions_file=settings.eval_questions_file,
        output_dir=settings.eval_results_dir,
    )

if __name__ == "__main__":
    asyncio.run(main())