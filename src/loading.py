import sys
import time
import shutil
from typing import Iterator, Sequence


def ft_tqdm(lst: Sequence[int]) -> Iterator[int]:
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.
    """

    start_time = time.time()
    terminal_width = shutil.get_terminal_size().columns
    max_length: int = terminal_width // 2
    total: int = len(lst)
    progress: int = 0
    progress_char = "="

    def render_progress() -> None:
        """Calculates percentage, items per seconds, elapsed/remaing time,"""
        """then prints the loading bar with its status on stdout."""

        elapsed_time: float = time.time() - start_time
        avg_time_per_item: float = elapsed_time / progress if progress > 0 else 0
        remaining_time: float = avg_time_per_item * (total - progress)
        items_per_sec: float = progress / elapsed_time if elapsed_time > 0 else 0

        percent_unit: float = progress / total
        filled_length: int = round(percent_unit * (max_length - 1))

        elapsed: str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        remaining: str = time.strftime("%M:%S", time.gmtime(remaining_time))

        progress_bar: str = (progress_char * filled_length + ">").ljust(max_length)
        status: str = f"{elapsed}<{remaining}, {items_per_sec:.2f}it/s"

        sys.stdout.write(
            f"\r\033[K{progress}/{total} epochs | [{progress_bar:{max_length}}] | "
            f"[{status}]"
        )
        sys.stdout.flush()

    for item in lst:
        yield item
        progress += 1

        if progress % 100 == 0 or progress == total:
            render_progress()
    sys.stdout.write("\n")
