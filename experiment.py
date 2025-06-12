import psynet.experiment
from psynet.bot import Bot
from psynet.modular_page import (
    ModularPage,
    PushButtonControl,
    ImagePrompt,
    Prompt,
    DropdownControl,
    SurveyJSControl,
    SliderControl,
    HtmlSliderControl
)
from psynet.timeline import Module, Timeline, Response
from psynet.trial.chain import ChainNode

from psynet.trial.static import StaticNode, StaticTrialMaker, StaticTrial
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.consent import MainConsent

from psynet.asset import CachedAsset, LocalStorage, S3Storage

from markupsafe import Markup

from os import listdir
from os.path import basename

import pandas as pd
import numpy as np

import json

from typing import List

MODE = "HOTAIR"

assert MODE in ["HOTAIR", "PROLIFIC"]

N_EXPERTISE_TRIALS = 3
N_EXPERTISE_NODES = 50

N_TARGET_TRIPLETS_PER_PARTICIPANTS = 150
N_MAX_TRIPLETS_PER_PARTICIPANTS = 150
N_TRIALS_PER_TRIPLET = 5

N_TARGET_RATINGS_PER_PARTICIPANTS = 50
N_MAX_RATINGS_PER_PARTICIPANTS = 50
N_TRIALS_PER_RATING = 5


def thompson_sampling(nodes: List[psynet.trial.static.StaticNode]) -> int:
    """Retrieve an informative expertise-assessment task
    by performing online Thompson sampling.
    The reward associated with a task
    is the F-score of the prediction of experts vs non-experts,
    based on prior participants' answers.

    Args:
        nodes (List[psynet.trial.static.StaticNode]): candidate nodes

    Returns:
        int: id of the selected node
    """
    alpha_prior = 1
    beta_prior = 1

    successes, failures = (
        {False: dict(), True: dict()},
        {False: dict(), True: dict()},
    )
    rewards = dict()

    for node in nodes:
        for trial in node.viable_trials:
            if trial.var.expert_status is None:
                continue

            is_expert = trial.var.expert_status

            if trial.var.successful_prediction == True:
                successes[is_expert][node.id] = successes[is_expert].get(node.id, 0) + 1
            elif trial.var.successful_prediction == False:
                failures[is_expert][node.id] = failures[is_expert].get(node.id, 0) + 1

        accuracy = np.zeros(2)
        for is_expert in [False, True]:
            accuracy[1 if is_expert else 0] = np.random.beta(
                alpha_prior + successes[is_expert].get(node.id, 0),
                beta_prior + failures[is_expert].get(node.id, 0),
            )

        # F-score
        rewards[node.id] = 2 * np.prod(accuracy) / np.sum(accuracy)

    best_node = sorted(
        list(rewards.keys()),
        key=lambda node: rewards[node],
        reverse=True,
    )[0]
    return best_node


def build_aesthetic_comparison_nodes():
    triplets_locations = [
        "/Users/lucasgautheron/Documents/cs/tasks/clip",
        "/Users/lucasgautheron/Documents/cs/tasks/random"
    ]

    for block, location in enumerate(triplets_locations):
        images = [f"{location}/{f}" for f in listdir(location) if f.endswith(".png")]

        nodes = [
            StaticNode(
                definition={
                    "id": basename(image),
                    "hashes": basename(image).split(".")[0].split("_"),
                },
                assets={
                    "stimulus": CachedAsset(image),
                },
                block=f"{block}",
            )
            for image in images
        ]

    return nodes


def build_aesthetic_rating_nodes():
    triplets_locations = [
        "/Users/lucasgautheron/Documents/cs/tasks/rating1",
    ]

    for block, location in enumerate(triplets_locations):
        images = [f"{location}/{f}" for f in listdir(location) if f.endswith(".png")]

        nodes = [
            StaticNode(
                definition={
                    "id": basename(image),
                    "hash": basename(image).split(".")[0],
                },
                assets={
                    "stimulus": CachedAsset(image),
                },
                block=f"{block}",
            )
            for image in images
        ]

    return nodes


class ExpertiseNode(ChainNode):
    def summarize_trials(self, trials: list, experiment, participant):
        answers = np.array([trial.answer["reproduce"] for trial in trials])
        print(answers)
        return trials


def build_expertise_nodes(n: int):
    triplets_location = "/Users/lucasgautheron/Documents/cs/tasks/guess-image-title/"
    tasks = pd.read_csv(
        "/Users/lucasgautheron/Documents/cs/tasks/guess-image-title.csv"
    ).head(n)

    nodes = [
        ExpertiseNode(
            definition={
                "id": task["image"],
                "title": task["target_title"],
                "choices": [task[f"choice_{i}"] for i in range(task["num_choices"])],
            },
            assets={"stimulus": CachedAsset(f"{triplets_location}/{task['image']}")},
        )
        for task in tasks.to_dict(orient="records")
    ]

    return nodes


aesthetic_comparison_nodes = build_aesthetic_comparison_nodes()
aesthetic_rating_nodes = build_aesthetic_rating_nodes()
expertise_nodes = build_expertise_nodes(N_EXPERTISE_NODES)


class ExpertiseTrial(StaticTrial):
    time_estimate = 60

    def __init__(self, experiment, node, participant, *args, **kwargs):
        super().__init__(experiment, node, participant, *args, **kwargs)

        # success variable is True if the answer to the trial
        # correctly predicts the expert-status of the participant
        self.var.successful_prediction = None
        self.var.expert_status = None

    def show_trial(self, experiment, participant):
        choices = self.definition["choices"]
        np.random.shuffle(choices)
        asset = self.assets["stimulus"]

        return ModularPage(
            "classify_diagram",
            ImagePrompt(
                asset,
                Markup("<div style='text-align: center; margin: 1em;'>Can you guess the title of the article from which this diagram was extracted?</div>"),
                width=350,
                height=350,
            ),
            DropdownControl(
                choices=choices + ["unknown"],
                labels=choices + ["I don't know!"],
                name="title_guess",
            ),
            bot_response=np.random.choice(choices + ["unknown"]),
        )


class CompareTrial(StaticTrial):
    time_estimate = 3

    def show_trial(self, experiment, participant):
        asset = self.assets["stimulus"]

        choices = [0, 1, 2]

        return ModularPage(
            "compare_diagrams",
            ImagePrompt(
                asset,
                Markup(
                    "<div style='text-align: center; margin: 1em;'>Which diagram is the prettiest among the three, in your opinion?</div>"),
                width=900,
                height=300,
            ),
            PushButtonControl(
                choices=choices,
                labels=["Left diagram", "Middle diagram", "Right diagram"],
                arrange_vertically=False,
            ),
            bot_response=np.random.choice(choices),
        )


class RateTrial(StaticTrial):
    time_estimate = 3

    def show_trial(self, experiment, participant):
        asset = self.assets["stimulus"]

        return ModularPage(
            "rate_diagrams",
            ImagePrompt(
                asset,
                Markup(
                    "<div style='text-align: center; margin: 1em;'>Please rate the diagram aesthetically on a scale from left (ugliest) to right (prettiest).</div>"),
                width=350,
                height=350,
            ),
            # PushButtonControl(
            #     choices=np.arange(10) + 1,
            #     labels=list(np.arange(10) + 1),
            #     arrange_vertically=False,
            # ),
            SliderControl(
                start_value=5,
                min_value=0,
                max_value=10,
                slider_id="slider",
                template_filename="slider_value.html"
            ),
            bot_response=np.random.uniform(1, 10),
        )


class ExpertiseTrialMaker(StaticTrialMaker):
    def custom_network_filter(self, candidates, participant):
        nodes = []
        for candidate in candidates:
            nodes += candidate.nodes()

        next_node_id = thompson_sampling(nodes)

        for candidate in candidates:
            if any([node.id == next_node_id for node in candidate.nodes()]):
                return [candidate]

        return candidates

    def finalize_trial(self, answer, trial, experiment, participant):
        # Get participant expertise
        response = Response.query.filter_by(
            question="survey", participant_id=participant.id
        ).one()

        expert = response.answer.get("expertise") in [
            "I am a computer scientist",
            "I am a scientist, but in another STEM field",
        ]

        node = trial.node

        correct_answer = answer == node.definition["title"]
        successful_prediction = (correct_answer == True and expert == True) or (
                (correct_answer == False) and (expert == False)
        )

        trial.var.successful_prediction = successful_prediction == True
        trial.var.expert_status = expert

        super().finalize_trial(answer, trial, experiment, participant)


expertise_trial = ExpertiseTrialMaker(
    id_="expertise_trial",
    trial_class=ExpertiseTrial,
    nodes=expertise_nodes,
    expected_trials_per_participant=N_EXPERTISE_TRIALS,
    max_trials_per_participant=N_EXPERTISE_TRIALS,
    target_trials_per_node=None,
    target_n_participants=1,
    recruit_mode="n_participants",
    allow_repeated_nodes=False,
    balance_across_nodes=False,
)


class AestheticTrialMaker(StaticTrialMaker):
    def choose_block_order(self, experiment, participant, blocks):
        return sorted(blocks)


aesthetic_comparison_trial = AestheticTrialMaker(
    id_="aesthetic_comparison_trial",
    trial_class=CompareTrial,
    nodes=aesthetic_comparison_nodes,
    expected_trials_per_participant=N_TARGET_TRIPLETS_PER_PARTICIPANTS,
    max_trials_per_participant=N_MAX_TRIPLETS_PER_PARTICIPANTS,
    target_trials_per_node=N_TRIALS_PER_TRIPLET,
    target_n_participants=1,
    recruit_mode="n_participants",
    allow_repeated_nodes=False,
    balance_across_nodes=False,
    n_repeat_trials=3
)

aesthetic_rating_trial = AestheticTrialMaker(
    id_="aesthetic_rating_trial",
    trial_class=RateTrial,
    nodes=aesthetic_rating_nodes,
    expected_trials_per_participant=N_TARGET_RATINGS_PER_PARTICIPANTS,
    max_trials_per_participant=N_MAX_RATINGS_PER_PARTICIPANTS,
    target_trials_per_node=N_TRIALS_PER_RATING,
    target_n_participants=1,
    recruit_mode="n_participants",
    allow_repeated_nodes=False,
    balance_across_nodes=False,
    n_repeat_trials=3
)

survey = ModularPage(
    "survey",
    "Before you start, please tell us a bit more about yourself!",
    SurveyJSControl(
        {
            "pages": [
                {
                    "name": "survey",
                    "elements": [
                        {
                            "type": "radiogroup",
                            "name": "expertise",
                            "title": "Which of these best describes you?",
                            "isRequired": "true",
                            "choices": [
                                "I am a computer scientist",
                                "I am a scientist, but in another STEM field",
                                "I am not a scientist",
                            ],
                        },
                        {
                            "type": "radiogroup",
                            "name": "education",
                            "title": "What is your highest degree?",
                            "isRequired": "true",
                            "choices": [
                                "High-school degree or less",
                                "BA/BSc or equivalent",
                                "MA/MSc or equivalent",
                                "PhD",
                            ],
                            "showOtherItem": "true",
                            "showNoneItem": "true",
                        },
                        {
                            "type": "boolean",
                            "name": "tikz",
                            "title": "Are you a user of TikZ, a programming language primarily used by scientists to produce scientific diagrams?",
                            "isRequired": "true",
                        },
                    ],
                }
            ],
            "headerView": "advanced",
        }
    ),
    time_estimate=15,
    bot_response=lambda: {
        "expertise": "I am not a scientist",
        "education": "BA/BSc",
        "tikz": False,
    },
)


def _(s):
    return s

def get_prolific_settings():
    with open("pt_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))
    return {
        "recruiter": "prolific",
        "base_payment": 2.12,
        "prolific_estimated_completion_minutes": 15,
        "prolific_recruitment_config": qualification,
        "auto_recruit": False,
        "wage_per_hour": 0,
        "currency": "£",
        "show_reward": False,
    }
recruiter_settings = get_prolific_settings()

class Exp(psynet.experiment.Experiment):
    label = "Pretty diagrams"
    asset_storage = LocalStorage()
    # asset_storage = LocalStorage("assets")
    # asset_storage = S3Storage("psynet-tests", "diagrams-aesthetics")


    config = {
        "recruiter": MODE.lower(),
        "wage_per_hour": 9,
        # "publish_experiment": False,
        "title": _(
            "Pretty diagrams (Chrome browser, ~15 minutes to complete, ~2pounds)"),

        "description": " ".join([
            _("This experiment requires you to rate images (scientific diagrams) according to how pretty you find them."),
            _("We recommend opening the experiment in an incognito window in Chrome, as some browser add-ons can interfere with the experiment."),
            _("If you have any questions or concerns, please contact us through Prolific.")
        ]),

        'initial_recruitment_size': 10,
        "auto_recruit": False,
        "show_reward": False,
        "contact_email_on_error": "lucas.gautheron@gmail.com",
        "organization_name": "Max Planck Institute for Empirical Aesthetics"
    }

    if MODE == "PROLIFIC":
        config.update(**recruiter_settings)

    timeline = Timeline(
        MainConsent(),
        survey,
        InfoPage(
            Markup(
                "Before we begin, let us try to assess your familiarity with the scientific domain in question very briefly."
            ),
            time_estimate=1,
        ),
        expertise_trial,
        InfoPage(
            Markup("Fantastic, we can now start the aesthetic judgment task!"), time_estimate=1
        ),
        aesthetic_comparison_trial,
        InfoPage(
            Markup(
                "Thank you! Let us finish with a slightly different task, assessing your aesthetic preferences in an other way."),
            time_estimate=1
        ),
        aesthetic_rating_trial,
        SuccessfulEndPage(),
    )

    test_n_bots = 10

    def test_check_bot(self, bot: Bot, **kwargs):
        assert not bot.failed
        trials = bot.all_trials

        n_target_trials = N_EXPERTISE_TRIALS + N_TARGET_TRIPLETS_PER_PARTICIPANTS + N_TARGET_RATINGS_PER_PARTICIPANTS
        assert len(trials) == n_target_trials, f"{len(trials)} != {n_target_trials}"
        assert all([t.complete for t in trials])
        assert all([t.finalized for t in trials])
