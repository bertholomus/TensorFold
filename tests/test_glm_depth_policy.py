"""GLM-5.3's draft depth by acceptance (decode.AcceptPolicy, TF_GLM_DEPTH_COST), on the CPU: drafts accepted every
round keep the deepest chain, drafts rejected at once fall to one, an in-between rate picks the depth with the most
expected tokens per cost (shallower when several streams share a round: a draft row adds its experts' weight reads
against a smaller share of the round), and every few rounds one probe goes a position deeper."""

from __future__ import annotations


def _policy(most=3, cost=0.2):
    from tensorfold.families.glm_moe_dsa.cuda.decode import AcceptPolicy

    return AcceptPolicy(most, cost=cost)


def test_kept_drafts_keep_the_deepest_chain():
    p = _policy()
    depths = [p.next(3, 3) for _ in range(40)]
    assert set(depths) == {3}


def test_rejected_drafts_fall_to_one_but_probe_deeper():
    from tensorfold.families.glm_moe_dsa.cuda.decode import DEPTH_PROBE

    p = _policy()
    depths = [p.next(1, 0) for _ in range(64)]
    tail = depths[32:]
    assert min(tail) == 1 and max(tail) == 2                     # the probes, one deeper
    assert sum(d == 2 for d in tail) == len(tail) // DEPTH_PROBE


def test_an_in_between_rate_weighs_expected_tokens_against_cost():
    p = _policy(cost=0.2)
    p.a = [0.55, 0.45, 0.4]
    p.rounds = 1                                                # off the probe rounds
    assert p.best() == 1                                        # 1.55 / 1.2 against 1.80 / 1.4 and 1.90 / 1.6
    p.rounds = 1
    assert p.best(0.05) == 3                                    # a cheap draft row: the whole chain


def test_streams_sharing_a_round_make_draft_rows_dearer():
    from tensorfold.families.glm_moe_dsa.cuda.decode import shared_cost

    assert shared_cost(0.2, 1) == 0.2
    assert abs(shared_cost(0.2, 4) - 0.5) < 1e-12               # 4 x 0.2 / (1 + 3 x 0.2)
    assert shared_cost(0.2, 2) < shared_cost(0.2, 3) < shared_cost(0.2, 4) < 1.0
    p = _policy(cost=0.2)
    p.a = [0.7, 0.7, 0.7]
    p.rounds = 1
    alone = p.best(shared_cost(p.cost, 1))                      # 2.53 / 1.6 beats 2.19 / 1.4 and 1.7 / 1.2
    p.rounds = 1
    four = p.best(shared_cost(p.cost, 4))                       # at 0.5 a row: 1.7 / 1.5 beats 2.19 / 2.0, 2.53 / 2.5
    assert alone == 3 and four == 1


def test_only_the_rounds_that_reached_a_position_move_it():
    p = _policy()
    p.update(3, 1)                                              # first kept, second rejected, third never verified
    assert p.a[0] > 0.8 and p.a[1] < 0.8 and p.a[2] == 0.8
