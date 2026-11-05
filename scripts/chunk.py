#!/usr/bin/env python3
"""
Chunk cleaned markdown files using both strategies.
Outputs chunks to data/processed/{fixed,heading}/<source>/.
"""

from hybrid_rag.ingestion.pipeline import main

if __name__ == "__main__":
    main()