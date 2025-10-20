import psynet.experiment
from psynet.bot import Bot
from psynet.demography.general import (
    BasicDemography,
)
from psynet.modular_page import (
    ModularPage,
    PushButtonControl,
    ImagePrompt,
)

from psynet.timeline import Timeline, CodeBlock, ModuleState, Response, Event
from psynet.trial.static import StaticNode, StaticTrialMaker, StaticTrial
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.consent import MainConsent, NoConsent

from psynet.asset import (
    CachedAsset, S3Storage,
    asset,
)

from markupsafe import Markup

from os.path import basename, splitext

import pandas as pd

S3_BUCKET = "lucasgautheron"
S3_KEY = "diagrams-aesthetics"

def get_s3_url(stimulus):
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_KEY}/{stimulus}"


DEBUG = False
MODE = "HOTAIR"

DURATION_ESTIMATE = 15 * 60
N_EXPERTISE_TRIALS = 3
N_EXPERTISE_NODES = 50

N_TARGET_TRIPLETS_PER_PARTICIPANTS = 15 if DEBUG else 150
N_MAX_TRIPLETS_PER_PARTICIPANTS = 15 if DEBUG else 150
N_TRIALS_PER_TRIPLET = 5

N_TARGET_RATINGS_PER_PARTICIPANTS = 5 if DEBUG else 75
N_MAX_RATINGS_PER_PARTICIPANTS = 5 if DEBUG else 75
N_TRIALS_PER_RATING = 5


class ClassificationTrial(StaticTrial):
    time_estimate = 3

    def show_trial(self, experiment, participant):
        asset = self.assets["stimulus"]

        return ModularPage(
            "classify_diagrams",
            ImagePrompt(
                asset,
                Markup(
                    "<div style='text-align: center; margin: 1em;'>Please flag junk diagrams.</div>",
                ),
                width=350,
                height=350,
            ),
            PushButtonControl(
                choices=[0, 1],
                labels=["Ok", "Junk!"],
                arrange_vertically=False,
            ),
            bot_response=0,
        )


class ClassificationTrialMaker(StaticTrialMaker):
    def __init__(self, *args, **kwargs):
        tasks_metadata = ["static/tasks/junk-random.csv"]

        nodes = []
        for block, location in enumerate(tasks_metadata):
            tasks = pd.read_csv(location).to_dict(orient="records")
            images_location = splitext(basename(location))[0]

            nodes += [
                StaticNode(
                    definition={
                        "id": basename(task["hash0"]),
                        "hash": task["hash0"],
                    },
                    assets={
                        "stimulus": asset(
                            get_s3_url(f"{images_location}/{task['hash0']}.png"),
                            cache=True,
                        ),
                    },
                    block=f"{block}",
                )
                for task in tasks
            ]

        super().__init__(*args, **kwargs, nodes=nodes)

    def choose_block_order(self, experiment, participant, blocks):
        return sorted(blocks)


classification_trial = ClassificationTrialMaker(
    id_="aesthetic_comparison_trial",
    trial_class=ClassificationTrial,
    expected_trials_per_participant=1000,
    max_trials_per_participant=1000,
    target_trials_per_node=1,
    target_n_participants=1,
    recruit_mode="n_participants",
    allow_repeated_nodes=False,
    balance_across_nodes=False,
    n_repeat_trials=0,
)

class Exp(psynet.experiment.Experiment):
    label = "Pretty diagrams"
    # asset_storage = LocalStorage()
    # asset_storage = LocalStorage    # asset_storage = LocalStorage("assets")("assets")
    asset_storage = S3Storage(S3_BUCKET, S3_KEY)

    config = {
        "wage_per_hour": 0,
        # "publish_experiment": False,
        "title": (
            "Pretty diagrams (Chrome browser, ~15 minutes to complete, ~2£)",
        ),

        "description": " ".join(
            [
                "..."
            ],
        ),

        'initial_recruitment_size': 10,
        "auto_recruit": False,
        "show_reward": False,
        "contact_email_on_error": "lucas.gautheron@gmail.com",
        "organization_name": "Max Planck Institute for Empirical Aesthetics",
    }

    timeline = Timeline(
        NoConsent() if DEBUG else MainConsent(),
        classification_trial,
        SuccessfulEndPage(),
    )

    test_n_bots = 10

    def test_check_bot(self, bot: Bot, **kwargs):
        assert not bot.failed
        trials = bot.all_trials

        n_target_trials = 100
        assert len(
            trials,
        ) == n_target_trials, f"{len(trials)} != {n_target_trials}"
        assert all([t.complete for t in trials])
        assert all([t.finalized for t in trials])
