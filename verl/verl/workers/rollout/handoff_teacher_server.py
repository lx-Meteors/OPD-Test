# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone vLLM process used for Teacher continuation generation.

The Actor and Reward roles can live in the same Ray worker process. vLLM sleep
mode only permits one engine per process, so the frozen Teacher engine must live
in a separate OS process. The two engines still time-share the same GPU: the
Student engine is asleep while this service generates, and this service sleeps
before returning its result to the Reward worker.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any

_AUTHKEY = b"verl-handoff-teacher"
_DISTRIBUTED_ENV_KEYS = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "DIST_INIT_METHOD",
}


class TeacherVLLMClient:
    """Own and communicate with one standalone Teacher vLLM process."""

    def __init__(
        self,
        *,
        model_path: str,
        rank: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        enforce_eager: bool,
        trust_remote_code: bool,
    ) -> None:
        self.model_path = model_path
        self.rank = rank
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.trust_remote_code = trust_remote_code
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("Teacher gpu_memory_utilization must be between 0 and 1.")
        if max_model_len <= 0 or max_num_seqs <= 0:
            raise ValueError("Teacher max_model_len and max_num_seqs must be positive.")
        if max_num_batched_tokens < max_model_len:
            raise ValueError(
                "Teacher max_num_batched_tokens must be at least max_model_len when chunked prefill is enabled."
            )
        self.process: subprocess.Popen | None = None
        self.connection: Connection | None = None
        self.socket_path = f"/tmp/verl_handoff_teacher_{os.getpid()}_{rank}.sock"
        atexit.register(self.close)

    def _recv(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        if self.connection is None or self.process is None:
            raise RuntimeError("Teacher vLLM service has not been started.")
        started_at = time.monotonic()
        while not self.connection.poll(1.0):
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"Teacher vLLM process exited unexpectedly with code {return_code}.")
            if timeout_seconds is not None and time.monotonic() - started_at > timeout_seconds:
                raise TimeoutError(f"Teacher vLLM service did not become ready within {timeout_seconds:.0f} seconds.")
        message = self.connection.recv()
        if message.get("status") == "error":
            raise RuntimeError(f"Teacher vLLM service failed:\n{message['error']}")
        return message

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None and self.connection is not None:
            return

        self.close()
        socket_file = Path(self.socket_path)
        socket_file.unlink(missing_ok=True)
        listener = Listener(self.socket_path, family="AF_UNIX", authkey=_AUTHKEY)
        listener._listener._socket.settimeout(60)  # noqa: SLF001 - multiprocessing exposes no accept timeout

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--socket-path",
            self.socket_path,
            "--model-path",
            self.model_path,
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--seed",
            str(self.rank),
        ]
        if self.enforce_eager:
            command.append("--enforce-eager")
        if self.trust_remote_code:
            command.append("--trust-remote-code")

        child_env = os.environ.copy()
        for key in _DISTRIBUTED_ENV_KEYS:
            child_env.pop(key, None)

        try:
            self.process = subprocess.Popen(command, env=child_env)
            self.connection = listener.accept()
        except BaseException:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            self.process = None
            Path(self.socket_path).unlink(missing_ok=True)
            raise
        finally:
            listener.close()

        try:
            initializing = self._recv(timeout_seconds=30)
            if initializing.get("status") != "initializing":
                raise RuntimeError(f"Unexpected Teacher vLLM startup message: {initializing}")
            ready = self._recv(timeout_seconds=900)
            if ready.get("status") != "ready":
                raise RuntimeError(f"Unexpected Teacher vLLM startup message: {ready}")
        except BaseException:
            self.close()
            raise

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[dict[str, Any]]:
        if not prompt_token_ids:
            return []
        self._start()
        assert self.connection is not None
        request = {
            "op": "generate",
            "prompt_token_ids": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            self.connection.send(request)
            response = self._recv()
        except (BrokenPipeError, EOFError, OSError):
            # A stale child should not permanently poison a long training run.
            self.close()
            self._start()
            assert self.connection is not None
            self.connection.send(request)
            response = self._recv()
        if response.get("status") != "ok":
            raise RuntimeError(f"Unexpected Teacher vLLM response: {response}")
        return response["outputs"]

    def close(self) -> None:
        connection, process = self.connection, self.process
        self.connection = None
        self.process = None
        if connection is not None:
            try:
                connection.send({"op": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        Path(self.socket_path).unlink(missing_ok=True)


def _terminate_with_parent() -> None:
    """Ask Linux to terminate this service if its Ray worker parent dies."""

    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGTERM)
    except (ImportError, OSError):
        pass


def _serve(args: argparse.Namespace) -> None:
    _terminate_with_parent()
    connection = Client(args.socket_path, family="AF_UNIX", authkey=_AUTHKEY)
    connection.send({"status": "initializing"})
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=1,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_sleep_mode=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=args.enforce_eager,
            trust_remote_code=args.trust_remote_code,
            disable_log_stats=True,
            seed=args.seed,
        )
        connection.send({"status": "ready"})
        sleeping = False
        while True:
            request = connection.recv()
            operation = request.get("op")
            if operation == "shutdown":
                break
            if operation != "generate":
                raise ValueError(f"Unknown Teacher service operation: {operation!r}")
            if sleeping:
                llm.wake_up()
                sleeping = False

            temperature = float(request["temperature"])
            sampling_params = SamplingParams(
                n=1,
                max_tokens=int(request["max_tokens"]),
                temperature=max(temperature, 0.0),
                top_p=float(request["top_p"]) if temperature > 0 else 1.0,
                top_k=-1,
                detokenize=False,
            )
            prompts = [{"prompt_token_ids": list(token_ids)} for token_ids in request["prompt_token_ids"]]
            raw_outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            outputs = []
            for request_output in raw_outputs:
                sample = request_output.outputs[0]
                outputs.append(
                    {
                        "token_ids": list(sample.token_ids),
                        "finish_reason": str(sample.finish_reason) if sample.finish_reason is not None else None,
                    }
                )

            # Do not tell the parent generation is complete until Teacher GPU
            # weights and KV cache have been released for FSDP scoring.
            llm.sleep(level=1)
            sleeping = True
            connection.send({"status": "ok", "outputs": outputs})
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send({"status": "error", "error": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
        Path(args.socket_path).unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    _serve(_parse_args())
