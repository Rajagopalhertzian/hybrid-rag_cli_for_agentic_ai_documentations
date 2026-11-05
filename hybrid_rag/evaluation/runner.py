"""
Evaluation runner orchestrating the full eval pipeline.
"""

import asyncio
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import orjson
import yaml
from rich.console import Console
from rich.progress import Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer

from hybrid_rag.config import settings
from hybrid_rag.models import (
    ChunkerType,
    EvalQuestion,
    Evidence,
    Source,
    RetrievalMetrics,
    GenerationMetrics,
    EvalResult,
)
from hybrid_rag.evaluation.baseline import closed_book_answer
from hybrid_rag.evaluation.metrics import compute_retrieval_metrics, compute_generation_metrics
from hybrid_rag.generation.synthesizer import synthesize_answer
from hybrid_rag.retrieval.hybrid import hybrid_search
from hybrid_rag.retrieval.qdrant import get_qdrant_client

console = Console()


def load_questions(path: Path) -> list[EvalQuestion]:
    """Load evaluation questions from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    questions = []
    for item in data:
        evidence = [
            Evidence(
                url=e["url"],
                heading=e["heading"],
                char_range=tuple(e["char_range"]),
                source=Source(e.get("source", "mcp")),
            )
            for e in item.get("evidence", [])
        ]
        questions.append(
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                evidence=evidence,
                tags=item.get("tags", []),
            )
        )
    return questions


class EvaluationRunner:
    """Runs evaluation across chunkers and baseline."""

    def __init__(
        self,
        questions: list[EvalQuestion],
        output_dir: Optional[Path] = None,
    ):
        self.questions = questions
        self.output_dir = output_dir or settings.eval_results_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )

    async def run(self) -> dict[str, list[EvalResult]]:
        """Run full evaluation."""
        results = {
            "fixed": [],
            "heading": [],
            "baseline": [],
        }

        client = await get_qdrant_client()

        for chunker in [ChunkerType.FIXED, ChunkerType.HEADING]:
            console.print(f"\n[bold cyan]Evaluating {chunker.value} chunker...[/bold cyan]")
            chunker_results = await self._evaluate_chunker(chunker, client)
            results[chunker.value] = chunker_results

        console.print("\n[bold cyan]Evaluating closed-book baseline...[/bold cyan]")
        baseline_results = await self._evaluate_baseline()
        results["baseline"] = baseline_results

        # Save results
        self._save_results(results)
        self._print_summary(results)

        return results

    async def _evaluate_chunker(
        self,
        chunker: ChunkerType,
        client,
    ) -> list[EvalResult]:
        """Evaluate a single chunker across all questions."""
        results = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[green]{chunker.value}", total=len(self.questions))

            for question in self.questions:
                # Embed query
                query_vector = self.embedding_model.encode(
                    [question.question],
                    normalize_embeddings=True,
                )[0].tolist()

                # Retrieve
                retrieved = await hybrid_search(
                    query_text=question.question,
                    query_vector=query_vector,
                    source=None,
                    chunker=chunker,
                    client=client,
                )

                # Generate
                answer = await synthesize_answer(question.question, retrieved)

                # Compute metrics
                retrieval_metrics = compute_retrieval_metrics(retrieved, question.evidence)
                generation_metrics = await compute_generation_metrics(
                    question.question,
                    answer.text,
                    retrieved,
                )

                result = EvalResult(
                    question_id=question.id,
                    chunker_type=chunker,
                    retrieval=retrieval_metrics,
                    generation=generation_metrics,
                    answer=answer,
                )
                results.append(result)
                progress.advance(task)

        return results

    async def _evaluate_baseline(self) -> list[EvalResult]:
        """Evaluate closed-book baseline."""
        results = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]baseline", total=len(self.questions))

            for question in self.questions:
                # Generate without retrieval
                answer = await closed_book_answer(question.question)

                # Compute generation metrics (no retrieval metrics)
                generation_metrics = await compute_generation_metrics(
                    question.question,
                    answer.text,
                    [],
                )

                # Empty retrieval metrics
                retrieval_metrics = RetrievalMetrics(
                    recall_at_5=0.0,
                    recall_at_10=0.0,
                    mrr=0.0,
                    ndcg_at_10=0.0,
                )

                result = EvalResult(
                    question_id=question.id,
                    chunker_type=ChunkerType.FIXED,  # Placeholder
                    retrieval=retrieval_metrics,
                    generation=generation_metrics,
                    answer=answer,
                )
                results.append(result)
                progress.advance(task)

        return results

    def _save_results(self, results: dict[str, list[EvalResult]]):
        """Save results to JSON and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = self.output_dir / f"eval_{timestamp}.json"
        json_data = {
            "timestamp": timestamp,
            "results": {
                k: [
                    {
                        "question_id": r.question_id,
                        "chunker_type": r.chunker_type.value,
                        "retrieval": {
                            "recall_at_5": r.retrieval.recall_at_5,
                            "recall_at_10": r.retrieval.recall_at_10,
                            "mrr": r.retrieval.mrr,
                            "ndcg_at_10": r.retrieval.ndcg_at_10,
                        },
                        "generation": {
                            "faithfulness": r.generation.faithfulness,
                            "answer_relevancy": r.generation.answer_relevancy,
                            "context_precision": r.generation.context_precision,
                            "context_recall": r.generation.context_recall,
                        },
                        "answer": {
                            "text": r.answer.text,
                            "citations": r.answer.citations,
                            "latency_ms": r.answer.latency_ms,
                            "model": r.answer.model,
                        },
                    }
                    for r in v
                ]
                for k, v in results.items()
            },
        }
        json_path.write_bytes(orjson.dumps(json_data, option=orjson.OPT_INDENT_2))

        # CSV summary
        csv_path = self.output_dir / f"eval_{timestamp}_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "chunker",
                "question_id",
                "recall@5",
                "recall@10",
                "mrr",
                "ndcg@10",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "latency_ms",
            ])
            for chunker, chunker_results in results.items():
                for r in chunker_results:
                    writer.writerow([
                        chunker,
                        r.question_id,
                        f"{r.retrieval.recall_at_5:.4f}",
                        f"{r.retrieval.recall_at_10:.4f}",
                        f"{r.retrieval.mrr:.4f}",
                        f"{r.retrieval.ndcg_at_10:.4f}",
                        f"{r.generation.faithfulness:.4f}",
                        f"{r.generation.answer_relevancy:.4f}",
                        f"{r.generation.context_precision:.4f}",
                        f"{r.generation.context_recall:.4f}",
                        f"{r.answer.latency_ms:.0f}",
                    ])

        # Latest symlink
        latest_json = self.output_dir / "latest.json"
        latest_csv = self.output_dir / "latest_summary.csv"
        for p in [latest_json, latest_csv]:
            if p.exists():
                p.unlink()
            p.symlink_to(json_path.name if p == latest_json else csv_path.name)

        console.print(f"\n[green]Results saved to {json_path} and {csv_path}[/green]")

    def _print_summary(self, results: dict[str, list[EvalResult]]):
        """Print summary table."""
        table = Table(title="Evaluation Summary")
        table.add_column("Chunker", style="cyan")
        table.add_column("Recall@5", justify="right")
        table.add_column("Recall@10", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("nDCG@10", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Ans. Relevancy", justify="right")

        for chunker, chunker_results in results.items():
            if not chunker_results:
                continue

            n = len(chunker_results)
            avg_recall5 = sum(r.retrieval.recall_at_5 for r in chunker_results) / n
            avg_recall10 = sum(r.retrieval.recall_at_10 for r in chunker_results) / n
            avg_mrr = sum(r.retrieval.mrr for r in chunker_results) / n
            avg_ndcg = sum(r.retrieval.ndcg_at_10 for r in chunker_results) / n
            avg_faith = sum(r.generation.faithfulness for r in chunker_results) / n
            avg_rel = sum(r.generation.answer_relevancy for r in chunker_results) / n

            table.add_row(
                chunker,
                f"{avg_recall5:.3f}",
                f"{avg_recall10:.3f}",
                f"{avg_mrr:.3f}",
                f"{avg_ndcg:.3f}",
                f"{avg_faith:.3f}",
                f"{avg_rel:.3f}",
            )

        console.print(table)


async def run_evaluation(
    questions_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, list[EvalResult]]:
    """Main entry point for evaluation."""
    questions_path = questions_file or settings.eval_questions_file
    questions = load_questions(questions_path)
    console.print(f"Loaded {len(questions)} evaluation questions")

    runner = EvaluationRunner(questions, output_dir)
    return await runner.run()