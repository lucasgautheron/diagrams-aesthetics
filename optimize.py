import cvxpy as cp

C = 30000  # target comparison trials
R = 15000  # target ratings

N_PARTICIPANTS = 200  # target amount of participant

C_TIME_PER_TRIAL = 3.5
R_TIME_PER_TRIAL = 3.5
FIXED_TIME = 60 * 4

comparisons_per_participant = cp.Variable(integer=True)
ratings_per_participant = cp.Variable(integer=True)

objective = cp.Minimize(
    N_PARTICIPANTS * FIXED_TIME
    + C_TIME_PER_TRIAL * N_PARTICIPANTS * comparisons_per_participant
    + R_TIME_PER_TRIAL * N_PARTICIPANTS * ratings_per_participant
)
constraints = [
    comparisons_per_participant * N_PARTICIPANTS >= C,
    ratings_per_participant * N_PARTICIPANTS >= R,
]
prob = cp.Problem(objective, constraints)

result = prob.solve()
print("Budget ($):", 12 * result / 60 / 60)
print(
    "Time per participant (minutes):",
    (
        FIXED_TIME
        + C_TIME_PER_TRIAL * comparisons_per_participant.value
        + R_TIME_PER_TRIAL * ratings_per_participant.value
    )
    / 60,
)
print(comparisons_per_participant.value)
print(ratings_per_participant.value)
