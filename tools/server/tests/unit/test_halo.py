# halo-hybrid: server behaviour this tree adds (rejection sampling of drafts at T > 0, the flags' environment aliases).
# CPU-sized models, like the rest of this directory.
import re

import pytest
from utils import *

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 0
    server.spec_draft_n_max = 4
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


@pytest.mark.parametrize("spec_rs", ["1", "0"])
def test_sampled_draft_follows_grammar(spec_rs, monkeypatch):
    # sampled requests verify drafts with rejection sampling (LLAMA_SPEC_RS=1, the default) or by comparing tokens
    # (=0); with a grammar the accepted tokens must still be ones the grammar allows
    global server
    monkeypatch.setenv("LLAMA_SPEC_RS", spec_rs)
    server.start()
    for seed in (1, 2, 3):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "temperature": 0.8,
            "seed": seed,
            "n_predict": 24,
            "grammar": 'root ::= ("yes" | "no") (" " [a-z]{1,8}){1,3} "."',
        })
        assert res.status_code == 200, res.body
        assert re.fullmatch(r"(yes|no)( [a-z]{1,8}){1,3}\.", res.body["content"]), res.body["content"]


def test_sampled_draft_drafts():
    # the draft path still runs for sampled requests (rejection sampling accepts some drafts)
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.7,
        "seed": 42,
        "n_predict": 48,
    })
    assert res.status_code == 200, res.body
    assert res.body["timings"]["draft_n"] > 0
    assert 0 <= res.body["timings"]["draft_n_accepted"] <= res.body["timings"]["draft_n"]


def test_env_aliases_of_the_halo_flags(monkeypatch):
    # --spec-draft-ubatch-size, --spec-draft-n-max-sampled, --prefill-lanes and --qsa-chunk-mb read these when the
    # flag is absent; on one CPU device the lanes stay off and the QSA chunking has nothing to chunk
    global server
    monkeypatch.setenv("LLAMA_SPEC_DRAFT_UB", "64")
    monkeypatch.setenv("LLAMA_SPEC_NMAX_SAMPLED", "1")
    monkeypatch.setenv("LLAMA_PREFILL_LANES", "2")
    monkeypatch.setenv("LLAMA_QSA_CHUNK_MB", "256")
    server.n_ubatch = 512
    server.log_path = "./tmp/test_halo_env_aliases.log"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "n_predict": 16,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 16
    log = open(server.log_path).read()
    assert "draft n_ubatch = 64" in log   # the LLAMA_SPEC_DRAFT_UB alias reached the draft context
    assert "LLAMA_PREFILL_LANES=2" in log and "LLAMA_QSA_CHUNK_MB=256" in log   # the startup environment line


def test_env_alias_rejects_a_bad_value(monkeypatch):
    # a value the flag would refuse is refused from the environment too, instead of being ignored
    global server
    monkeypatch.setenv("LLAMA_SPEC_DRAFT_UB", "-5")
    with pytest.raises(Exception):
        server.start(timeout_seconds=30)


def test_prefill_lanes_zero_is_off(monkeypatch):
    # LLAMA_PREFILL_LANES=0 has always meant "off"; the alias must keep accepting it
    global server
    monkeypatch.setenv("LLAMA_PREFILL_LANES", "0")
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "temperature": 0.0, "n_predict": 4})
    assert res.status_code == 200, res.body


def test_kvu_full_pool_purges_the_lru_idle_slot():
    # -kvu -np 2 with idle slots kept resident (--no-cache-idle-slots): when the shared pool cannot take a new prompt,
    # the least recently used idle slot leaves it for the RAM prompt cache, and its conversation comes back from there
    global server
    server = ServerPreset.stories15m_moe()
    server.kv_unified = True
    server.n_slots = 2
    server.n_ctx = 256   # the pool; the model's own context is 256 too
    server.no_cache_idle_slots = True
    server.cache_ram = 64
    server.log_path = "./tmp/test_halo_kvu_purge.log"
    server.start()
    a = "The quick brown fox jumps over the lazy dog. " * 12
    b = "A little girl went to the park and played with her red ball. " * 10

    def complete(prompt):
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "temperature": 0.0, "n_predict": 16})
        assert res.status_code == 200, res.body
        return res.body

    a1 = complete(a)
    b1 = complete(b)
    assert a1["tokens_evaluated"] + b1["tokens_evaluated"] + 32 > 256   # the two do not fit the pool together
    a2 = complete(a)
    assert a2["content"] == a1["content"]
    assert a2["timings"]["cache_n"] > 0   # restored from the prompt cache, not prefilled again
    log = open(server.log_path).read()
    assert "purging slot" in log
