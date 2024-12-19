# WILT: Wason Inductive Logic Test

**THIS REPO IS OUTMODED**

The new repo for this project can be found at [https://github.com/riotgames/wilt](https://github.com/riotgames/wilt).

<p align="center">
    <img src="https://github.com/ambisinister/wilt/blob/main/docs/wilt.png?raw=true" width="128">
</p>

**WILT** a simple, general, multi-turn inductive logic test for language models. The test is comprised of simple black-box functions of three variables, where you may test up to 30 examples in order to observe the input-output relationships of these functions.

This test is based on the [Wason 2-4-6 task](https://journals.sagepub.com/doi/10.1080/17470216008416717). By asking the model to infer functions involving three variables, we get a reasoning benchmark which is both robust to memorization and reliably predictive of a model's ability to solve simple problems based on previous observations. 

## Usage

```
python main.py --model="llama3-70b-8192"
```

To see the supported model strings, please consult models/model_factory.py. This repo is being actively worked on, so new models are added occasionally and not enumerated here as a result.

### Constrained Generation

Local models that can be run on my 3090 will also use [Guidance](https://github.com/guidance-ai/guidance) to constrain their generation to adhere to a format which can be parsed easily by the testing harness. For the most part, these models do not perform very well -- the use of guidance just allows the models to complete the test properly rather than failing to engage with the test at all.

## Initial results on Light Test Split

Initial results on the light split (10 questions only) for a few promient models (GPT-4o, Claude Sonnet 3.5, etc) can be found in [the initial blogpost](https://planetbanatt.net/articles/wason.html) for the first version of this project. I find the light split to be a good show of reasoning capabilities as a back-of-the-napkin test; not necessarily enough to differentiate similar models, but enough to roughly assess a model's ability to do simple reasoning.

## Initial results on Full Test Set

Coming Soon
