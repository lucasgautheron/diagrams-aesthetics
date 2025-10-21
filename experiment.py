import psynet.experiment
from psynet.bot import Bot
from psynet.demography.general import (
    Gender, Age, FormalEducation,
)
from psynet.modular_page import (
    ModularPage,
    PushButtonControl,
    ImagePrompt,
    Prompt,
    DropdownControl,
    SurveyJSControl,
    SliderControl,
)

from psynet.timeline import Timeline, CodeBlock, ModuleState, Response, Event
from psynet.participant import Participant
from psynet.trial.chain import ChainNode

from psynet.trial.main import Trial
from psynet.trial.static import StaticNode, StaticTrialMaker, StaticTrial
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.consent import MainConsent, NoConsent

from psynet.asset import (
    CachedAsset, S3Storage,
    asset,
)

from psynet.utils import log_time_taken

from dallinger import db
from sqlalchemy import Column, Integer
from sqlalchemy.ext.declarative import declared_attr

from markupsafe import Markup

from os.path import basename, splitext

import pandas as pd
import numpy as np
import random

import json
import csv

from typing import List, Union, Optional

from psynet.utils import get_logger

S3_BUCKET = "cap-lucasgautheron"
S3_KEY = "diagrams-aesthetics/tasks"

def get_s3_url(stimulus):
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_KEY}/{stimulus}"

logger = get_logger()

DEBUG = False
MODE = "HOTAIR"

TIMELINE = "AESTHETIC"

if TIMELINE == "EXPERTISE":
    N_EXPERTISE_TRIALS = 30
    N_EXPERTISE_NODES = 60
else:
    N_EXPERTISE_TRIALS = 3
    N_EXPERTISE_NODES = 20

N_TARGET_TRIPLETS_PER_PARTICIPANTS = 15 if DEBUG else 80
N_MAX_TRIPLETS_PER_PARTICIPANTS = 15 if DEBUG else 80
N_TRIALS_PER_TRIPLET = 5

N_TARGET_RATINGS_PER_PARTICIPANTS = 5 if DEBUG else 40
N_MAX_RATINGS_PER_PARTICIPANTS = 5 if DEBUG else 40
N_TRIALS_PER_RATING = 5

N_REPEAT_TRIALS = 2

DURATION_ESTIMATE = 60 + N_EXPERTISE_TRIALS*60 + N_TARGET_TRIPLETS_PER_PARTICIPANTS*11 + N_TARGET_RATINGS_PER_PARTICIPANTS*11


class ActiveInference:
    def get_optimal_node(self, nodes_ids, participant, data):
        z_i = participant.var.z

        S = 1000

        rewards = dict()
        eig = dict()
        utility = dict()
        p_outcome = dict()

        alphas = dict()
        betas = dict()

        z_participants = np.array(
            [
                data["participants"][participant_id]["z"]
                for participant_id in data["participants"]
                if data["participants"][participant_id]["z"] != None
            ],
        )

        alpha_z = 1 + np.sum(z_participants == 1)
        beta_z = 1 + np.sum(z_participants == 0)
        p_z = alpha_z / (alpha_z + beta_z)

        for node_id in nodes_ids:
            alpha = np.ones(2)
            beta = np.ones(2)

            for trial_id, trial in data["nodes"][node_id].items():
                if trial["y"] == True:
                    alpha[trial["z"]] += 1
                elif trial["y"] == False:
                    beta[trial["z"]] += 1

            alphas[node_id] = alpha
            betas[node_id] = beta

            alpha = alpha[:, np.newaxis]
            beta = beta[:, np.newaxis]

            phi = np.random.beta(
                alpha,
                beta,
                (2, S),
            )

            y = np.random.binomial(
                np.ones((2, S), dtype=int),
                phi,
                size=(
                    2,
                    S,
                ),
            )

            p_y_given_phi = phi * y + (1 - phi) * (1 - y)
            p_y = alpha / (alpha + beta) * y + beta / (alpha + beta) * (1 - y)

            EIG = np.mean(np.log(p_y_given_phi[z_i] / p_y[z_i]))

            gamma = 0.1
            p_z = 0.5  # assume equi-representation of experts and non-experts
            U = gamma * np.mean(
                p_z * y[1]
                + (1 - p_z) * (1 - y[0])
                - p_z * (1 - y[1])
                - (1 - p_z) * y[0],
            )

            rewards[node_id] = EIG + U
            eig[node_id] = EIG
            utility[node_id] = U
            p_outcome[node_id] = float((alpha / (alpha + beta))[z_i].mean())

        best_node = sorted(
            list(rewards.keys()),
            key=lambda node_id: rewards[node_id],
            reverse=True,
        )[0]

        if len(nodes_ids) == 15:
            with open(f"output/utility.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        rewards[best_node],
                        eig[best_node],
                        utility[best_node],
                    ],
                )

        return best_node, {0: 1 - p_outcome[best_node], 1: p_outcome[best_node]}


class ExpertiseNode(ChainNode):
    pass


class ExpertiseTrial(StaticTrial):
    time_estimate = 30

    @declared_attr
    def y(cls):
        return cls.__table__.c.get("y", Column(Integer))

    @declared_attr
    def z(cls):
        return cls.__table__.c.get("z", Column(Integer))

    def __init__(self, experiment, node, participant, *args, **kwargs):
        super().__init__(experiment, node, participant, *args, **kwargs)

    def show_trial(self, experiment, participant):
        choices = self.definition["choices"]
        np.random.shuffle(choices)
        asset = self.assets["stimulus"]

        return ModularPage(
            "classify_diagram",
            ImagePrompt(
                asset,
                Markup(
                    "<div style='text-align: center; margin: 1em;'>Can you guess the title of the article from which this diagram was extracted?</div>",
                ),
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

    def show_feedback(self, experiment, participant):
        if self.answer == "unknown":
            return None

        return InfoPage(
            Markup("Congratulations, this is correct!")
            if self.y == True
            else Markup("Nice try, but no :("),
        )


class ExpertiseTrialMaker(StaticTrialMaker):
    def __init__(self, *args, n_nodes, setup: str, **kwargs):
        triplets_location = ""
        tasks = pd.read_csv(
            "static/tasks/guess-image-title.csv",
        ).head(n_nodes)

        images_location = \
            splitext(basename("static/tasks/guess-image-title.csv"))[0]

        nodes = [
            ExpertiseNode(
                definition={
                    "id": task["image"],
                    "title": task["target_title"],
                    "choices": [task[f"choice_{i}"] for i in
                                range(task["num_choices"])],
                },
                assets={
                    "stimulus": asset(
                        get_s3_url(f"{images_location}/{task['image']}"),
                        cache=True,
                    ),
                },
            )
            for task in tasks.to_dict(orient="records")
        ]

        super().__init__(*args, **kwargs, nodes=nodes)

        self.optimizer = ActiveInference()

    def finalize_trial(self, answer, trial, experiment, participant):
        correct_answer = answer == trial.node.definition["title"]

        trial.y = int(correct_answer)

        z = participant.var.get("z", None)
        trial.z = int(z) if z is not None else None

        super().finalize_trial(answer, trial, experiment, participant)

    @log_time_taken
    def prior_data(self):
        data = {"nodes": dict(), "participants": dict()}

        # List participants involved in this trial maker
        participants = (
            db.session.query(Participant)
            .join(Participant._module_states)
            .filter(
                ModuleState.module_id == self.id, ModuleState.started == True,
            )
            .distinct()
            .all()
        )

        data["participants"] = {
            participant.id: {
                "z": (
                    participant.var.get("z", None)
                ),
            }
            for participant in participants
        }

        # Fetch all nodes related to this trial maker
        nodes = self.network_class.query.filter_by(
            trial_maker_id=self.id, full=False, failed=False,
        ).all()

        # Fetch all trials that belong to this trial maker
        trials = Trial.query.filter(
            Trial.failed == False,
            Trial.finalized == True,
            Trial.is_repeat_trial == False,
            Trial.trial_maker_id == self.id,
        ).all()

        trials_by_node = {}
        for trial in trials:
            if trial.node_id not in trials_by_node:
                trials_by_node[trial.node_id] = []
            trials_by_node[trial.node_id].append(trial)

        # Process trials for each node
        for node in nodes:
            data["nodes"][node.id] = {}

            if node.id in trials_by_node:
                data["nodes"][node.id] = {
                    trial.id: {
                        "y": trial.y,
                        "z": trial.z,
                        "participant_id": trial.participant_id,
                    }
                    for trial in trials_by_node[node.id]
                    if trial.y is not None
                }

        return data

    # overload the default prioritize_networks method
    @log_time_taken
    def prioritize_networks(self, networks, participant, experiment):
        if self.optimizer is None:
            return networks

        node_network = {
            network.head.id: i for i, network in enumerate(networks)
        }

        # retrieve all relevant prior data
        data = self.prior_data()

        # retrieve optimal node
        next_node, p = self.optimizer.get_optimal_node(
            list(node_network.keys()), participant, data,
        )

        # store the optimizer estimate of p(y)
        participant.var.set("p_y", p)

        # early-stopping
        if next_node is None:
            return []

        return [networks[node_network[next_node]]]


class AestheticComparisonPrompt(Prompt):
    def __init__(
            self,
            assets: List[CachedAsset],
            text: Union[str, Markup],
            width: str,
            height: str,
            show_after: float = 0.0,
            hide_after: Optional[float] = None,
            margin_top: str = "0px",
            margin_bottom: str = "0px",
            text_align: str = "left",
    ):
        super().__init__(text=text, text_align=text_align)

        self.urls = [asset.url for asset in assets]
        self.width = width
        self.height = height
        self.show_after = show_after
        self.hide_after = hide_after
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom

    macro = "comparison"
    external_template = "aesthetic-comparison.html"

    @property
    def metadata(self):
        return {
            "text": str(self.text),
            "urls": self.urls,
            "show_after": self.show_after,
            "hide_after": self.hide_after,
        }

    def update_events(self, events):
        events["promptStart"] = Event(
            is_triggered_by="trialStart", delay=self.show_after,
        )

        if self.hide_after is not None:
            events["promptEnd"] = Event(
                is_triggered_by="promptStart", delay=self.hide_after,
            )


class CompareTrial(StaticTrial):
    time_estimate = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hashes = self.definition["hashes"].copy()
        random.shuffle(hashes)
        self.var.hashes = hashes

    def show_trial(self, experiment, participant):
        hashes = self.var.hashes
        assets = [self.assets[hash] for hash in hashes]
        choices = self.var.hashes

        return ModularPage(
            "compare_diagrams",
            AestheticComparisonPrompt(
                assets,
                Markup(
                    "<div style='text-align: center; margin: 1em;'>Which diagram is the prettiest among the three, in your opinion?</div>",
                ),
                width=300,
                height=300,
            ),
            PushButtonControl(
                choices=choices,
                labels=["Left diagram", "Middle diagram", "Right diagram"],
                arrange_vertically=False,
            ),
            bot_response=np.random.choice(choices),
        )


class AestheticComparisonTrialMaker(StaticTrialMaker):
    def __init__(self, *args, **kwargs):
        triplets_locations = [
            "static/tasks/comparison-clusters-1.csv",
            "static/tasks/comparison-random-2.csv",
        ]

        nodes = []
        for block, location in enumerate(triplets_locations):
            tasks = pd.read_csv(location)
            images_location = splitext(basename(location))[0]

            nodes += [
                StaticNode(
                    definition={
                        "id": "_".join(
                            [task["hash0"], task["hash1"], task["hash2"]],
                        ),
                        "hashes": [task["hash0"], task["hash1"], task["hash2"]],
                    },
                    assets={
                        hash: asset(
                            get_s3_url(f"{images_location}/{hash}.png"),
                            cache=True,
                        )
                        for hash in
                        [task["hash0"], task["hash1"], task["hash2"]]
                    },
                )
                for task in tasks.to_dict(orient="records")
            ]

        super().__init__(*args, **kwargs, nodes=nodes)

    # def choose_block_order(self, experiment, participant, blocks):
    #     return sorted(blocks)


class RateTrial(StaticTrial):
    time_estimate = 10

    def show_trial(self, experiment, participant):
        asset = self.assets["stimulus"]

        return ModularPage(
            "rate_diagrams",
            ImagePrompt(
                asset,
                Markup(
                    "<div style='text-align: center; margin: 1em;'>Please rate the diagram aesthetically on a scale from left (ugliest) to right (prettiest).</div>",
                ),
                width=350,
                height=350,
            ),
            SliderControl(
                start_value=4,
                min_value=1,
                max_value=7,
                snap_values=np.arange(1, 7 + 1).tolist(),
                n_steps=7,
                slider_id="slider",
                template_filename="slider_value.html",
                minimal_time=2,
                minimal_interactions=0
            ),
            bot_response=np.random.randint(1, 7 + 1),
        )


class AestheticRatingTrialMaker(StaticTrialMaker):
    def __init__(self, *args, **kwargs):
        tasks_metadata = ["static/tasks/rating-1.csv"]

        nodes = []
        for block, location in enumerate(tasks_metadata):
            tasks = pd.read_csv(location).drop_duplicates("hash0").to_dict(orient="records")
            images_location = splitext(basename(location))[0]

            nodes += [
                StaticNode(
                    definition={
                        "id": basename(task["hash0"]),
                        "hash": task["hash0"],
                    },
                    assets={
                        "stimulus": asset(
                            get_s3_url(
                                f"{images_location}/{task['hash0']}.png",
                            ),
                            cache=True,
                        ),
                    },
                    block=f"{block}",
                )
                for task in tasks
            ]

        super().__init__(*args, **kwargs, nodes=nodes)

    #
    # def choose_block_order(self, experiment, participant, blocks):
    #     return sorted(blocks)

    def prioritize_networks(self, networks, participant, experiment):
        rate_trials = self.trial_class.query.filter_by(
            participant_id=participant.id, finalized=True
        ).all()

        if len(rate_trials) < 30:
            return networks

        rate_hashes = []
        for trial in rate_trials:
            rate_hashes.append(trial.definition["hash"])
        rate_hashes = set(rate_hashes)

        remaining_hashes = set([network.head.definition["hash"] for network in networks])
        assert len(remaining_hashes&rate_hashes) == 0
        available_hashes = remaining_hashes | rate_hashes

        compare_trials = aesthetic_comparison_trial.trial_class.query.filter_by(
            participant_id=participant.id, failed=False, finalized=True,
        )
        compare_hashes = []
        all_compare_hashes = set()
        n_completed_triplets = 0
        for trial in compare_trials:
            hashes = set(trial.definition["hashes"])
            compare_hashes.append(hashes)
            all_compare_hashes |= hashes
            if len(hashes&rate_hashes) == len(hashes):
                n_completed_triplets += 1

        logger.info(all_compare_hashes)

        logger.info(f"Number of completed triplets: {n_completed_triplets}")

        logger.info("RateTrial networks before filtering:")
        logger.info(len(networks))
        filtered_networks = [
            network for network in networks
            if network.head.definition["hash"] in all_compare_hashes
        ]
        logger.info("RateTrial networks after filtering:")
        logger.info(len(filtered_networks))

        if len(filtered_networks) == 0:
            return filtered_networks

        completable_triplets = [
            triplet for triplet in compare_hashes
            if len(triplet - available_hashes) == 0
            # All hashes in triplet are available
        ]

        if len(completable_triplets) <= n_completed_triplets:
            logger.info("No more completable triplets, return all networks.")
            return networks

        logger.info(f"Triplets before filtering: {len(compare_hashes)}")
        logger.info(f"Triplets after filtering: {len(completable_triplets)}")

        return sorted(
            filtered_networks,
            key=lambda network: (
                max(
                    [
                        # Score higher when we'd complete or nearly complete a triplet
                        len(triplet & rate_hashes)
                        if network.head.definition["hash"] in triplet and
                           network.head.definition["hash"] not in rate_hashes
                        else -1
                        for triplet in completable_triplets
                    ]
                ) if len(completable_triplets) > 0 else -1
            ),
            reverse=True,
        )


expertise_trial = ExpertiseTrialMaker(
    id_="expertise_trial",
    trial_class=ExpertiseTrial,
    n_nodes=N_EXPERTISE_NODES,
    setup="adaptive" if TIMELINE == "EXPERTISE" else "static",
    expected_trials_per_participant=N_EXPERTISE_TRIALS,
    max_trials_per_participant=N_EXPERTISE_TRIALS,
    target_trials_per_node=None,
    target_n_participants=1,
    recruit_mode="n_participants",
    allow_repeated_nodes=False,
    balance_across_nodes=False,
)

if TIMELINE != "EXPERTISE":
    aesthetic_comparison_trial = AestheticComparisonTrialMaker(
        id_="aesthetic_comparison_trial",
        trial_class=CompareTrial,
        expected_trials_per_participant=N_TARGET_TRIPLETS_PER_PARTICIPANTS,
        max_trials_per_participant=N_MAX_TRIPLETS_PER_PARTICIPANTS,
        target_trials_per_node=N_TRIALS_PER_TRIPLET,
        target_n_participants=1,
        recruit_mode="n_participants",
        allow_repeated_nodes=False,
        balance_across_nodes=False,
        n_repeat_trials=N_REPEAT_TRIALS,
    )

    aesthetic_rating_trial = AestheticRatingTrialMaker(
        id_="aesthetic_rating_trial",
        trial_class=RateTrial,
        expected_trials_per_participant=N_TARGET_RATINGS_PER_PARTICIPANTS,
        max_trials_per_participant=N_MAX_RATINGS_PER_PARTICIPANTS,
        target_trials_per_node=N_TRIALS_PER_RATING,
        target_n_participants=1,
        recruit_mode="n_participants",
        allow_repeated_nodes=False,
        balance_across_nodes=False,
        n_repeat_trials=N_REPEAT_TRIALS,
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
                                "I have a college degree in Science, Technology, Engineering, or Mathematics (STEM)",
                                "I have a college degree in another field",
                                "I do not have a college degree",
                            ],
                        },
                        {
                            "type": "boolean",
                            "name": "scientist",
                            "title": "Are you a scientist (e.g., working in academia, research institute, or industry)?",
                            "isRequired": "true",
                        },
                        {
                            "type": "boolean",
                            "name": "latex",
                            "title": "Are you a user of LaTeX, a typesetting language often used by researchers to write scientific documents?",
                            "isRequired": "true",
                        },
                    ],
                },
            ],
            "headerView": "advanced",
        },
    ),
    time_estimate=30,
    bot_response=lambda: {
        "expertise": "I am not a scientist",
        "education": "BA/BSc",
        "tikz": False,
    },
)


def _(s):
    return s


def get_prolific_settings(experiment_duration):
    with open("pt_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))

    return {
        "recruiter": "prolific",
        "base_payment": 9 * DURATION_ESTIMATE / 3600,
        "prolific_estimated_completion_minutes": DURATION_ESTIMATE / 60,
        "prolific_recruitment_config": qualification,
        "auto_recruit": False,
        "wage_per_hour": 0,
        "currency": "$",
        "show_reward": False,
    }


def get_cap_settings(experiment_duration):
    raise {
        "wage_per_hour": 12,
    }


assert MODE in ["HOTAIR", "PROLIFIC", "CAP"]

recruiters = {
    "HOTAIR": "hotair",
    "PROLIFIC": "prolific",
    "CAP": "cap-recruiter",
}

recruiter_settings = None
if MODE == "PROLIFIC":
    recruiter_settings = get_prolific_settings(DURATION_ESTIMATE)
elif MODE == "CAP":
    recruiter_settings = get_cap_settings(DURATION_ESTIMATE)


class Exp(psynet.experiment.Experiment):
    label = "Pretty diagrams"
    asset_storage = S3Storage(S3_BUCKET, S3_KEY)
    test_n_bots = 10

    config = {
        "recruiter": recruiters[MODE],
        "wage_per_hour": 0,
        # "publish_experiment": False,
        "title": _(
            "Pretty diagrams (Chrome browser, ~25 minutes to complete, ~3.6$)",
        ),

        "description": " ".join(
            [
                _(
                    "This experiment requires you to assess the visual appeal of several images (scientific diagrams).",
                ),
                _(
                    "We recommend opening the experiment in an incognito window in Chrome, as some browser add-ons can interfere with the experiment.",
                ),
                _(
                    "If you have any questions or concerns, please contact us through Prolific.",
                ),
            ],
        ),

        'initial_recruitment_size': 3,
        "auto_recruit": False,
        "show_reward": False,
        "contact_email_on_error": "lucas.gautheron@gmail.com",
        "organization_name": "Max Planck Institute for Empirical Aesthetics",
    }

    if MODE != "HOTAIR":
        config.update(**recruiter_settings)

    if TIMELINE == "EXPERTISE":
        timeline = Timeline(
            NoConsent() if DEBUG else MainConsent(),
            Gender(),
            FormalEducation(),
            survey,
            CodeBlock(
                lambda participant: participant.var.set(
                    "z", Response.query.filter_by(
                        question="survey", participant_id=participant.id,
                    ).one().answer.get("expertise") in [
                             "I have a college degree in Science, Technology, Engineering, or Mathematics (STEM)",
                         ],
                ),
            ),
            InfoPage(
                Markup(
                    f"<h3>Before we begin...</h3>"
                    f"<div style='margin: 10px;'>You will be shown a series of diagrams. For each diagram, you will have to guess the title of the scientific publication from which they originate, among multiple choices.</div>"
                    f"<div style='margin: 10px;'>If you have no idea, please say 'I don't know'. There is no reward or penalty for being right or wrong!</div>"
                    f"<div style='margin: 10px;'>If you make a guess, we will tell you whether you were correct or not.</div>"
                    f"<div style='border: 2px black; margin: 10px;'><img src='/static/images/task1.png' width='480' /></div>",
                ),
                time_estimate=5,
            ),
            expertise_trial,
            SuccessfulEndPage(),
        )

    else:
        timeline = Timeline(
            NoConsent() if DEBUG else MainConsent(),
            Gender(),
            FormalEducation(),
            survey,
            CodeBlock(
                lambda participant: participant.var.set(
                    "z", Response.query.filter_by(
                        question="survey", participant_id=participant.id,
                    ).one().answer.get("expertise") in [
                             "I have a college degree in Science, Technology, Engineering, or Mathematics (STEM)",
                         ],
                ),
            ),
            InfoPage(
                Markup(
                    f"<h3>Before we begin...</h3>"
                    f"<div style='margin: 10px;'>Before we begin, let us try to assess your familiarity with the scientific domain in question very briefly!</div>"
                    f"<div style='margin: 10px;'>You will be presented with a series of diagrams. For each diagram, you will have to guess the title of the scientific publication from which they originate, among multiple choices.</div>"
                    f"<div style='margin: 10px;'>If you have no idea, you may say 'I don't know'. There is no reward or penalty for being right or wrong!</div>"
                    f"<div style='border: 2px black; margin: 10px;'><img src='/static/images/task1.png' width='480' /></div>",
                ),
                time_estimate=5,
            ),
            expertise_trial,
            InfoPage(
                Markup(
                    f"<h3>Compare diagrams!</h3>"
                    f"<div style='margin: 10px;'>Fantastic, we can now start the aesthetic judgment task!</div>"
                    f"<div style='margin: 10px;'>You will be presented with a series of triplets of diagrams. For each triplet, you will have to pick the diagram that you find prettier.</div>"
                    f"<div style='border: 2px black; margin: 10px;'><img src='/static/images/task2.png' width='480' /></div>",
                ),
                time_estimate=5,
            ),
            aesthetic_comparison_trial,
            InfoPage(
                Markup(
                    f"<h3>Rate diagrams!</h3>"
                    "<div style='margin: 10px;'>Thank you! Let us finish with a slightly different task, assessing your aesthetic preferences in an other way.</div>"
                    f"<div style='margin: 10px;'>You will be presented with a series of diagrams. Rate each diagram from 1 (ugliest) to 7 (prettiest).</div>"
                    f"<div style='border: 2px black; margin: 10px;'><img src='/static/images/task3.png' width='480' /></div>",
                ),
                time_estimate=5,
            ),
            aesthetic_rating_trial,
            SuccessfulEndPage(),
        )

    def test_check_bot(self, bot: Bot, **kwargs):
        assert not bot.failed
        trials = bot.all_trials

        n_target_trials = N_EXPERTISE_TRIALS + N_TARGET_TRIPLETS_PER_PARTICIPANTS + N_TARGET_RATINGS_PER_PARTICIPANTS + 2 * N_REPEAT_TRIALS
        assert len(
            trials,
        ) == n_target_trials, f"{len(trials)} != {n_target_trials}"
        assert all([t.complete for t in trials])
        assert all([t.finalized for t in trials])
