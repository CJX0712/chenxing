"""Continuous batching scheduler invariants."""
from chenxing.core.sampler import SamplingParams
from chenxing.engine.scheduler import ContinuousBatchingScheduler
from chenxing.engine.sequence import Sequence, SequenceStatus


def test_fifo_admission_and_completion(tiny_engine, sample_prompt):
    sch = ContinuousBatchingScheduler(tiny_engine, max_running=4)
    seqs = [tiny_engine.create_sequence(sample_prompt, SamplingParams(max_tokens=3))
            for _ in range(3)]
    for s in seqs:
        sch.submit(s)
    done = sch.run_until_drained()
    assert len(done) == 3
    assert all(s.status == SequenceStatus.FINISHED for s in done)
    assert all(len(s.generated) == 3 for s in done)


def test_max_running_cap(tiny_engine, sample_prompt):
    sch = ContinuousBatchingScheduler(tiny_engine, max_running=1)
    for _ in range(3):
        sch.submit(tiny_engine.create_sequence(
            sample_prompt, SamplingParams(max_tokens=2)))
    assert sch.has_work
    sch.run_until_drained()
    assert not sch.has_work


def test_step_returns_finished(tiny_engine, sample_prompt):
    sch = ContinuousBatchingScheduler(tiny_engine, max_running=2)
    s1 = tiny_engine.create_sequence(sample_prompt, SamplingParams(max_tokens=1))
    s2 = tiny_engine.create_sequence(sample_prompt, SamplingParams(max_tokens=2))
    sch.submit(s1)
    sch.submit(s2)
    finished = sch.step()
    assert s1 in finished and s2 not in finished
    sch.run_until_drained()
    assert s2 in sch._finished
