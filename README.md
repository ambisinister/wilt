# WILT: Wason Inductive Logic Test

This repo contains a simple, general inductive logic test for language models. The test is comprised of simple black-box functions of three variables, where you may test up to 30 examples in order to observe the input-output relationships of these functions.

This test is based on the [Wason 2-4-6 task](https://journals.sagepub.com/doi/10.1080/17470216008416717). By procedurally generating simple functions involving three variables, we get a reasoning benchmark which is both robust to memorization and reliably predictive of a model's ability to solve simple problems based on previous observations.

## Usage

```
python eval.py
```