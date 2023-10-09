def delayed_linear_lambda(step, zero_till_step: int, increase_over_steps: int):
    if step <= zero_till_step:
        return 0.0
    elif step <= zero_till_step + increase_over_steps:
        return (step - zero_till_step) / increase_over_steps
    else:
        return 1.0
