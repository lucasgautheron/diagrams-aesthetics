# Tasks

In this experiment, participants rate scientific diagrams according to their aesthetic performance.

The experiment features three tasks:

1. A task for assessing participants' expertise about the target domain, in complement to their self-reported
   expert-status.
2. A comparison task in which participants choose the prettiest diagram among triplets
3. A direct rating task in which participants rate diagrams between 0 and 10.

# Task Comparison

| Aspect                       | Task 1 (Expertise) | Task 2 (Comparisons)                                 | Task 3 (Direct rating)             |
|------------------------------|--------------------|------------------------------------------------------|------------------------------------|
| **# Images**                 | 50                 | 3,000                                                | 3,000                              |
| **# Nodes**                  | 50                 | 6,000 triplets                                       | 3,000 images                       |
| **# Trials**                 |                    | 30,000 total (2x15,000 blocks), 5 trials per triplet | 15,000 total, 5 trials per triplet |
| **# Trials per Participant** | 3                  | 150 trials                                           | 75 trials                          |
| **# Participants Needed**    |                    | 100 for first block, 200 for both blocks             | 200 for completion                 

## Task 1 - Adaptive expertise assessment

![](static/images/task1.png)

## Task 2 - Subjective comparison

![](static/images/task2.png)

## Task 3 - Subjective rating

![](static/images/task3.png)


## S3 steps

```bash
aws s3 sync static/tasks s3://lucasgautheron/diagrams-aesthetics 
aws s3api put-bucket-policy --bucket lucasgautheron --policy file://my-policy.json
aws s3api put-bucket-cors --bucket lucasgautheron --cors-configuration file://my-cors.json
```
