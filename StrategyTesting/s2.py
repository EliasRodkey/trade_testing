from simulations import simulate_simset, compatible, signals


signal_args = {
    "signal_n" : range(5, 100, 10),
    "upper_bound" : range(70, 100, 10),
    "lower_bound" : range(10, 40, 10)
}
preference_args = {
    "preference_n" : range(5, 75, 10)
}

simulate_simset(
    signals.create_relative_strength_index_signals,
    signal_args,
    signals.calculate_rolling_sharpe_ratio,
    preference_args,
    contains_ma_type=True, 
    ma_types=compatible[2:],
    save_results=True
)