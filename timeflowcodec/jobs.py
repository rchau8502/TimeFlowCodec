"""Shared cooperative cancellation for codec and media jobs."""


class JobCancelled(Exception):
    """Raised at a safe checkpoint when a caller cancels a job."""


def check_cancel(cancel_check=None):
    if cancel_check is not None and cancel_check():
        raise JobCancelled("Job cancelled")


def report_progress(callback, phase, completed, total):
    if callback is not None:
        callback(phase, completed, total)
