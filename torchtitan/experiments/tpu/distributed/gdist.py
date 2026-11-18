"""Utility to programmatically expose torchrun API.

A trimmed down version to be used in OSS environments.
"""

import abc
import pathlib
import random
import shutil
import string
import sys
import tempfile

from absl import logging
import portpicker
import torch
from torch.distributed.launcher import api
from typing_extensions import override


def torchrun(fn, *, nnodes=None, nproc_per_node=None):
  """A programmatic API to `torchrun`, used to spawn `n` copies of `fn`.

  Here `fn` is the main entrypoint function to your PyTorch [distributed]
  program and `n = nnodes * nproc_per_node` [world_size].

  `torchrun` (https://pytorch.org/docs/stable/elastic/run.html) is the default
  launcher tool to start the worker processes when using distributed PyTorch.
  For those familiar with MPI, this is akin to `mpirun`.

  Use from the main program to spawn `nproc_per_node` copies of `fn` where
  `fn` is the main function of your PyTorch program. For multi-machine jobs,
  the main process on each machine should invoke `torchrun`. Effectively
  this runs `nnodes x nproc_per_node` copies of `fn`.

  Args:
    fn: The main function for the user training script.
    nnodes: The number of machines the job is running on. If not specified, it
      will be inferred from the runtime environment (defaulting to 1 for local
      and Colab runs).
    nproc_per_node: The number of local ranks to spawn. Defaults to the number
      of CUDA devices on the machine or 1 is no CUDA devices are found.

  Returns:
    A wrapped `fn` that can be called with the same args as `fn`
    but will launch multiple copies of `fn` and returns a dict
    of `{local_rank: fn()_on_local_rank}`.
  """

  def _run_deferred(*args):
    # this is required to defer any logic that we run until `absl.app.run`
    # has been invoked so that InitGoogle has had a chance to run and we
    # can do "normal" things like use absl.logging
    # (which will get swallowed into oblivion if InitGoogle hasn't run yet)
    environment = LocalEnvironment()
    log_dir = tempfile.mkdtemp(prefix="torchrun_")
    launch_config = environment.launch_config(
        nnodes=nnodes or environment.default_nnodes(),
        nproc_per_node=nproc_per_node or environment.default_nproc_per_node(),
        role=fn.__name__,
        log_dir=log_dir,
    )

    try:
      ret = api.elastic_launch(launch_config, environment.run(fn))(*args)
      if all(v is None for v in ret.values()):
        # if all the local ranks returned `None` this was probably
        # a `main()` method with a void return signature
        # since mains are typically invoked with absl.app.run() which
        # calls sys.exit(main()) we want to return `None` since sys.exit(None)
        # exits the program with exit code 0 (success). Otherwise, the program
        # will exit with a non-zero exit code (failure) despite the main
        # having run successfully.
        return None
      else:
        return ret
    finally:
      logging.info("Deleting temp log directory: %s", log_dir)
      shutil.rmtree(log_dir, ignore_errors=True)
      logging.info("Finished running `%s`", fn.__name__)

  return _run_deferred


class Environment(abc.ABC):
  """The environment this process is running in.

  This class is meant to be sub-classed to represent specific runtime
  environments. To get the current environment use

  >>> env = current_environment()

  Note: this class and its subclasses are meant to be used to create
    `LaunchConfig` for `torch.distributed.launcher.api.elastic_launch()`.
    Google users should prefer using `torch.google.distributed.torchrun()`
    rather than using this class to create `LaunchConfig` manually.
  """

  @abc.abstractmethod
  def run(self, fn):
    """Wraps `fn` w.r.t the environment.

    Usage:

    >>> def foo(name):
    ...   print(f"hello {name}")
    >>> current_env = current_environment()
    >>> current_env.run(foo)(name="kiuk")
    "hello kiuk"

    This is useful when environments need to do extra handling around
    `fn` to correctly run `fn`. Subclasses should override this
    method per their environment specs. Otherwise, the default
    simply returns the `fn` with no extra handling.

    For instance, in Colab, when `fn` is defined in a cell
    we need to wrap it in `DynamicDefBase` to make it run
    under multiprocessing. Therefore, `Colab` environment
    overrides this method to handle environment specific
    logic around `fn`.

    Args:
      fn: user function

    Returns:
      Wrapped `fn` that can be called the same way `fn` but wrapped
      with environment specific logic around `fn`.
    """
    pass

  @abc.abstractmethod
  def default_run_id(self):
    """The run_id to use for `torchrun`.

    Unique identifier for all machines participating in the job.

    Returns:
      A job-wide UID if the environment provides one. `None` otherwise
      (torchrun will create a random UUID).
    """
    pass

  @abc.abstractmethod
  def rdzv_endpoint(self):
    """The rdzv_endpoint to use for `torchrun`.

    For more information on what rendezvous is
    see: https://pytorch.org/docs/stable/elastic/rendezvous.html

    Returns:
      A tuple of `(hostname_ip_or_dns, port)` that can be used to reach
      machine_0. `hostname_ip_or_dns` should have a route from the rest
      of the machines in the job and `port` should be accessible.
      Note that technically, any machine (even one outside the job) can
      be chosen to host the `rdzv_endpoint`, but we typically use machine_0
      for consistency purposes.
    """
    pass

  @abc.abstractmethod
  def local_addr(self):
    """The address of the local node in rendezvous.

    For more information on what rendezvous is
    see: https://pytorch.org/docs/stable/elastic/rendezvous.html

    Returns:
      A string of hostname dns.
    """
    pass

  @abc.abstractmethod
  def default_nnodes(self):
    """Default number of machines in this job.

    In certain cases, the total number of machines in this job can be deduced.

    When this method returns `1`, it means that the job is running
    single-machine, multi-accelerator (e.g. multi-GPU).

    Returns:
      Actual number of machines in this job. 1 when this number cannot be
      deduced from the environment.
    """
    pass

  @abc.abstractmethod
  def name(self):
    """The human-readable name of this environment.

    Some environment sub-classes may represent multiple "physical" environments.
    """
    pass

  def default_nproc_per_node(self):
    """The default number of local ranks to run.

    Returns:
      If torch is compiled with CUDA support (e.g. built with `--config=cuda`)
      AND there is >0 GPUs on the machine, then returns the number of GPUs
      on the current machine. Otherwise, returns 1.
    """
    if torch.cuda.is_available():
      return torch.cuda.device_count()
    else:
      return 1

  def launch_config(self, **kwargs):
    """`LaunchConfig`s for this environment.

    Use `kwargs` to explicitly set `LaunchConfig` parameters.
    Otherwise, defaults will be assigned based on the current
    environment.

    Args:
      **kwargs: user overrides to default LaunchConfig values

    Returns:
      The `LaunchConfig` for `torch.distributed.launch.api.elastic_launch`
    """

    # nnodes is not a LaunchConfig param but we support it here for
    # convenience since in most cases min_nodes == max_nodes
    # and `torchrun --nnodes` supports it in CLI
    nnodes = kwargs.pop("nnodes", None)
    log_directory = kwargs.pop("log_dir", None)
    logs_specs = api.DefaultLogsSpecs(log_dir=log_directory)

    # assignment to "_" required to appease pylint (expression-not-assigned)
    _ = kwargs.setdefault("min_nodes", nnodes or self.default_nnodes())
    _ = kwargs.setdefault("max_nodes", nnodes or self.default_nnodes())
    _ = kwargs.setdefault("nproc_per_node", self.default_nproc_per_node())
    _ = kwargs.setdefault("run_id", self.default_run_id())

    task0_host, task0_port = self.rdzv_endpoint()
    _ = kwargs.setdefault("rdzv_endpoint", f"{task0_host}:{task0_port}")
    _ = kwargs.setdefault("local_addr", self.local_addr())
    _ = kwargs.setdefault("rdzv_backend", "c10d")
    _ = kwargs.setdefault("monitor_interval", 0.1)  # 100ms
    _ = kwargs.setdefault("max_restarts", 0)  # torchrun restarts
    _ = kwargs.setdefault("start_method", "spawn")
    _ = kwargs.setdefault("logs_specs", logs_specs)

    logging.info(
        "On `%s` creating torch.distributed.LaunchConfig in with:\n  %s\n",
        self.name(),
        "\n  ".join([f"{k:18s}: {v}" for k, v in kwargs.items()]),
    )
    return api.LaunchConfig(**kwargs)


class LocalEnvironment(Environment):
  """A "local" environment this process is running in.

  This class is not meant to be used directly, but instead
  provides environment-specific launch configurations to `torchrun`.
  Users looking to launch their main functions should use
  `torch.google.distributed.torchrun`.

  This class is meant to be sub-classed (e.g. ColabEnvironment) but itself
  represents a "local" environment.

  Local environments are those where the PyTorch program does not run
  multi-machine, e.g. local runs and single-machine CI test runs.
  """

  @override
  def run(self, fn):
    return fn

  @override
  def default_run_id(self):
    # `run_id` is used for machines (torchrun process on each machine)
    # to join machine_0 when running multi-machine distributed jobs
    # that is, `(rdzv_endpoint, run_id)` uniquely identify a "job"
    # to join.
    #
    # For local runs (aka single-machine, multi-local_rank)
    # we use `localhost:$random_free_port` as the rdzv_endpoint
    # since we don't expect any other machine to join the job
    # (e.g. all ranks are running in the same machine)
    # since a "free port" is chosen at runtime `rdzv_endpoint`
    # will uniquely identify a local job.
    # Therefore, we can simply use the main binary name (usually *.par)
    # without the extension suffixed by a short alpha-numeric random id

    binary_name = pathlib.Path(sys.argv[0]).stem
    random_id = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=4)
    )
    return f"{binary_name}_{random_id}"

  @override
  def rdzv_endpoint(self):
    return (self.local_addr(), portpicker.pick_unused_port())

  @override
  def local_addr(self):
    return "localhost"

  @override
  def default_nnodes(self):
    return 1

  @override
  def name(self) -> str:
    return "local"

