from cross_suite_generalization.build_goal_progress_review import select_windows, summarize


def episode(success):
    return dict(success=success, windows=[dict(width=20, end_action_index=i,
        path_m=.02, net_over_path=.2 if i == 30 else .8) for i in range(19,100)])


def test_review_selection_is_invariant_to_success_label():
    assert select_windows(episode(True)) == select_windows(episode(False))


def test_equal_exposure_excludes_short_episodes_and_future_windows():
    short = episode(True)
    short['windows'] = short['windows'][:10]
    a = episode(True)
    b = episode(False)
    for w in b['windows']:
        if w['end_action_index'] >= 60:
            w['net_over_path'] = 0
    result = summarize([a,b,short])
    assert result['True']['eligible_prefix60'] == 1
    assert result['True']['fraction_quantiles'] == result['False']['fraction_quantiles']
