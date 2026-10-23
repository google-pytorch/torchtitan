"""Streams the UniRef50 FASTA from UniProt into parquet shards.

Each shard has a ``text`` column with one protein sequence per row (plus the
UniRef ``ids``), the schema of the HF ``bloyal/uniref50`` mirror that the
``uniref50`` dataset entry in ``experiments/tpu/datasets.py`` expects.

Usage: python fasta_to_parquet.py <out_dir> <max_sequences> [--seqs_per_shard N]
"""

import argparse
from collections.abc import Iterable, Iterator
import gzip
import io
import itertools
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"


def read_fasta(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
  """Yields ``(id, sequence)`` pairs from FASTA lines."""
  seq_id, parts = None, []
  for line in lines:
    if line.startswith(">"):
      if seq_id is not None:
        yield seq_id, "".join(parts)
      seq_id, parts = line[1:].split(" ", 1)[0], []
    else:
      parts.append(line.strip())
  if seq_id is not None:
    yield seq_id, "".join(parts)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("out_dir")
  parser.add_argument("max_sequences", type=int)
  parser.add_argument("--seqs_per_shard", type=int, default=500_000)
  args = parser.parse_args()

  with subprocess.Popen(["curl", "-sL", URL], stdout=subprocess.PIPE) as proc:
    lines = io.TextIOWrapper(
        gzip.GzipFile(fileobj=proc.stdout), encoding="ascii", errors="replace"
    )
    records = itertools.islice(read_fasta(lines), args.max_sequences)
    for shard, batch in enumerate(itertools.batched(records, args.seqs_per_shard)):
      ids, seqs = zip(*batch)
      path = f"{args.out_dir}/train-{shard:05d}.parquet"
      pq.write_table(
          pa.table({"ids": list(ids), "text": list(seqs)}), path, compression="zstd"
      )
      print(f"wrote {path} ({len(ids)} seqs)", flush=True)
    proc.kill()


if __name__ == "__main__":
  main()
