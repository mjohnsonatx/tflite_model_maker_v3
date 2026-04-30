# EfficientDet Hyperparameter Tuning Guide

This guide is written for practical tuning on a **single-class EfficientDet Lite0 detector** like your dumbbell model.

It covers:
- what each major hyperparameter does
- when to adjust it
- what metric changes to expect
- concrete examples of useful and not-useful tuning moves

---

# 1) How to think about tuning

Before changing anything, decide which problem you are actually trying to solve.

## If AP50 is low
The model is missing the object or producing weak detections.

Most likely knobs:
- anchors / aspect ratios / num_scales
- learning rate
- score threshold at evaluation
- data quality / label quality
- augmentation strength

## If AP50 is high but AP75 is lagging
The model finds the object, but boxes are not tight enough.

Most likely knobs:
- box loss weight
- image size
- learning rate refinement
- anchor fit
- better box labels / curation

## If train loss keeps dropping but val/test stop improving
You are probably not getting much from more capacity or longer training.

Most likely knobs:
- regularization
- data curation
- harder validation split
- checkpoint selection metric

## If the model is unstable or converges inconsistently
Most likely knobs:
- learning rate
- warmup
- batch size
- mixed precision / environment issues

---

# 2) When to adjust alpha / gamma / delta

These are **not** the first knobs I would touch.

I usually adjust them only after:
- anchors are reasonable
- learning rate is in a sane range
- augmentation is not obviously too aggressive
- labels are fairly clean
- the model has already learned the object well enough to make the failure mode obvious

## Alpha
Use when the model is badly imbalanced between positives and negatives and focal loss is not weighting things well.

In a single-class detector, I usually leave this alone unless there is a strong reason.

## Gamma
Use when classification behavior is the real problem:
- too much focus on easy negatives
- too many false positives
- weak ranking of harder positives

Gamma changes how much focal loss downweights easy examples.

## Delta
Use when localization is the problem and you want to change how the box regression penalty behaves for small vs larger errors.

For your type of project, I would only move delta after:
- AP50 is already pretty good
- AP75 is the main bottleneck
- anchors and image size have already been addressed as much as practical

---

# 3) Hyperparameters by category

## A. Model / dataset structure

### `num_classes`
What it does:
- Sets the number of foreground classes.
- For your case, this is `1` for dumbbell.

When to change it:
- Only when the dataset taxonomy changes.

Likely effect:
- Not a tuning knob for performance in a fixed one-class setup.

Concrete example:
- Changing `num_classes` from `1` to `2` just because the model struggles would be wrong.
- Only do that if you actually add another labeled class.

---

### `image_size`
What it does:
- Sets the input resolution.
- Larger image size gives the detector more pixel detail.

When to change it:
- When AP50 is decent but AP75 is limited by localization detail.
- When the object can be small or visually thin.

Likely effect:
- Often improves AP75 more than AP50.
- Increases memory use and training cost.

Concrete example:
- If the detector sees the dumbbell but boxes are loose, increasing image size can help.
- If VRAM is already tight and batch size is capped, image size may be too expensive for now.

---

## B. Augmentation

### `input_rand_hflip`
What it does:
- Randomly flips training images horizontally.

When to change it:
- Usually keep it on for symmetric object detection unless left/right orientation matters semantically.

Likely effect:
- Better generalization with almost no downside in many detection tasks.

Concrete example:
- Dumbbells usually remain valid under horizontal flip, so this is a good default.

---

### `jitter_min`, `jitter_max`
What they do:
- Control random scale jitter during training.

When to change them:
- If training views are too uniform, mild jitter helps.
- If labels are noisy or the model is fragile, overly aggressive jitter can hurt.

Likely effect:
- Too little jitter: model becomes brittle.
- Too much jitter: unrealistic crops and worse localization.

Concrete example:
- `0.85–1.15` is a mild, sane range for a curated single-class problem.
- `0.6–1.4` can be too aggressive if the object framing becomes unrealistic.

---

### `autoaugment_policy`
What it does:
- Applies stronger automated augmentation policies.

When to change it:
- Mostly when the dataset is large and diverse enough to support stronger augmentation.

Likely effect:
- Can improve robustness.
- Can also hurt if the task is narrow and label quality is not excellent.

Concrete example:
- For a dumbbell detector with imperfect labels, `None` may outperform a strong policy because it preserves realism.

---

### `grid_mask`
What it does:
- Randomly masks grid regions of the image.

When to change it:
- Only if occlusion robustness is a real target and the dataset is large enough.

Likely effect:
- Can help robustness.
- Can easily hurt a small one-class dataset.

Concrete example:
- I would usually leave this off for your current setup.

---

## C. Anchors / detection geometry

### `aspect_ratios`
What it does:
- Defines the anchor box shapes.

When to change it:
- When the object has a characteristic shape or orientation that the default anchors fit poorly.

Likely effect:
- Better matching between anchors and ground truth.
- Can improve recall and box quality.

Concrete example:
- Adding `0.33` and `3.0` makes sense for a dumbbell because it can appear elongated.
- If the model improves after this change, it usually means anchor fit was part of the bottleneck.

---

### `anchor_scale`
What it does:
- Controls the base anchor size at each pyramid level.

When to change it:
- When the object scale distribution does not match the default anchors.

Likely effect:
- Helps object-anchor matching.
- Can improve AP if the defaults are too large or too small.

Concrete example:
- Moving from `4.0` to `2.5` can help if the object appears smaller relative to the frame than the default anchor assumption expects.

---

### `num_scales`
What it does:
- Sets how many anchor scales are used per feature level.

When to change it:
- When you want either broader scale coverage or a simpler, cleaner anchor set.

Likely effect:
- More scales: more coverage, more complexity.
- Fewer scales: simpler matching, sometimes better for a narrow single-object task.

Concrete example:
- Reducing from `3` to `2` can help a single-class problem by simplifying the search space.

---

## D. Classification loss

### `label_smoothing`
What it does:
- Softens the class targets before computing classification loss. In EfficientDet’s focal loss path, the code applies smoothing by blending targets toward `0.5`. That makes labels less “hard.”

When to change it:
- If the classifier is overconfident, brittle, or overfitting.
- If labels are a little noisy.

Likely effect:
- Higher smoothing: softer confidence, sometimes better generalization.
- Lower smoothing: sharper ranking, sometimes better AP if labels are clean.

Concrete example:
- Going from `0.1` to `0.05` can help a one-class detector become less mushy.
- Going from `0.05` to `0.02` or `0.0` can help only if labels are clean enough; otherwise it can make the model more brittle.

Practical note:
- In your runs, reducing smoothing too far did not help. That is a good sign that `0.05` is already in a reasonable range.

---

### `alpha`
What it does:
- In focal loss, alpha balances positive and negative examples.

When to change it:
- Only when class imbalance behavior is clearly off.
- Usually later in tuning, not early.

Likely effect:
- Higher alpha puts more weight on positive examples.
- Lower alpha puts relatively more weight on negatives.

Concrete example:
- If the model is missing true dumbbells too often and you suspect positives are underweighted, a small alpha increase might help.
- If the model is flooding predictions with false positives, raising alpha is probably the wrong move.

Practical recommendation:
- For your task, keep `alpha = 0.25` unless classification imbalance is clearly the issue.

---

### `gamma`
What it does:
- In focal loss, gamma controls how strongly easy examples are downweighted.
- Higher gamma focuses more on hard examples.

When to change it:
- When classification behavior is the bottleneck, not localization.

Likely effect:
- Higher gamma: more focus on hard examples, sometimes less stable.
- Lower gamma: less aggressive focusing, often more forgiving.

Concrete example:
- Dropping gamma from `1.6` to `1.25` can improve recall and confidence calibration when the detector was being too harsh on medium-difficulty positives.
- Raising gamma when AP50 is already high but AP75 is low usually will not be the best next move, because that is more of a localization issue.

Practical recommendation:
- Adjust gamma when false positives / false negatives / ranking are the main problem.
- Do not use gamma as your first fix for box quality.

---

## E. Localization loss

### `delta`
What it does:
- Sets the transition point in Huber loss between the quadratic region and the linear region.

When to change it:
- When box regression behavior seems to need a different penalty shape.
- Usually after anchors and learning rate are already sensible.

Likely effect:
- Lower delta: more errors fall into the linear region sooner.
- Higher delta: more moderate errors stay in the quadratic region longer.

Concrete example:
- Going from `0.05` to `0.1` can help if the model needs smoother, less harsh treatment of moderate box errors.
- If labels are noisy, delta changes can have limited upside because noisy boxes cap how much regression can improve.

Practical note:
- EfficientDet’s code comments say delta is typically around the mean regression target value, and gives an example around `[0.1, 0.1, 0.2, 0.2]` for a typical setup. That is one reason `0.1` is a sensible default neighborhood.

---

### `box_loss_weight`
What it does:
- Multiplies the box regression loss before adding it to classification loss.

When to change it:
- When the detector already finds the object well but boxes are not tight enough.

Likely effect:
- Higher box loss weight emphasizes localization more.
- Too high can hurt classification balance or make training less stable.

Concrete example:
- If AP50 is strong but AP75 lags, moving from `60` to `80` can help.
- If AP50 drops after raising box loss weight a lot, you may have overemphasized regression.

Practical recommendation:
- This is a very good knob when your problem is “finds it, but not tightly.”

---

## F. Learning rate and schedule

### `learning_rate`
What it does:
- Sets the base LR before EfficientDet adjusts it by effective batch size.

When to change it:
- Almost anytime convergence quality or stability is questionable.

Likely effect:
- Too high: unstable, noisy, weak refinement.
- Too low: slow learning, underfitting, weak exploration.

Concrete example:
- If a model learns the object but never tightens boxes well, a slightly lower LR can help refinement.
- If the model is still under-learning, too low an LR can waste epochs.

Practical note:
- EfficientDet scales learning rate by effective batch size relative to a default batch size. So changing batch size can change the effective LR even if the configured value stays the same.

---

### `lr_warmup_init`
What it does:
- Starting LR during warmup.

When to change it:
- Usually only if early training is unstable or warmup seems ineffective.

Likely effect:
- Lower warmup init can improve stability.
- Usually a secondary knob.

Concrete example:
- If training spikes or diverges very early, reducing warmup init can help.

---

### `lr_warmup_epoch`
What it does:
- Number of warmup epochs before the main LR schedule takes over.

When to change it:
- If early training is unstable or you changed batch size / optimizer behavior.

Likely effect:
- More warmup can stabilize the start of training.
- Too much warmup can delay useful learning.

Concrete example:
- If batch size is fixed and training is already stable, I would usually leave this alone.

---

### `lr_decay_method`
What it does:
- Chooses the LR schedule type.
- Common values are `cosine`, `stepwise`, `polynomial`, `constant`.

When to change it:
- When the overall training curve suggests the anneal behavior is wrong.

Likely effect:
- `cosine`: smooth decay across the whole run.
- `stepwise`: abrupt drops at chosen milestones.
- `polynomial`: steady curve controlled by `poly_lr_power`.

Concrete example:
- If you are using `cosine`, the model gets a smooth long refinement tail. That often fits fine-tuning well.

Practical note:
- Under EfficientDet’s code, `first_lr_drop_epoch` and `second_lr_drop_epoch` matter for `stepwise`, not for `cosine`.

---

### `first_lr_drop_epoch`, `second_lr_drop_epoch`
What they do:
- Milestones for the stepwise LR schedule.

When to change them:
- Only when you are actually using `stepwise`.

Likely effect:
- Earlier drops shift training into refinement sooner.

Concrete example:
- If your schedule is `cosine`, spending time tuning these is usually wasted effort.

---

### `poly_lr_power`
What it does:
- Controls the curve shape for polynomial decay.

When to change it:
- Only when `lr_decay_method='polynomial'`.

Likely effect:
- Higher or lower values change how quickly LR falls through the run.

Concrete example:
- If you use cosine, this does not matter.

---

### `num_epochs`
What it does:
- Total training duration.

When to change it:
- When the best checkpoint is landing too close to the end.

Likely effect:
- More epochs allow more refinement.
- Too many epochs waste time or overfit.

Concrete example:
- If best epoch is repeatedly around `180–200`, moving from `200` to `225` is reasonable.
- If best epoch is consistently around `120`, jumping to `250` is probably not the best first move.

---

## G. Optimization / regularization

### `optimizer`
What it does:
- Chooses the optimization algorithm.

When to change it:
- Only if you have a strong reason.

Likely effect:
- `sgd` is standard and stable for EfficientDet.
- `adam` may change behavior a lot and is not usually the first thing to try here.

Concrete example:
- I would not switch optimizers mid-tuning unless the current training is clearly failing.

---

### `momentum`
What it does:
- Controls the momentum term for SGD.

When to change it:
- Rarely.

Likely effect:
- Can affect smoothness and stability, but usually not the highest-value knob.

Concrete example:
- `0.9` is standard and usually fine.

---

### `weight_decay`
What it does:
- Adds L2 regularization on weights.

When to change it:
- When overfitting or underfitting behavior suggests the model is too unconstrained or too constrained.

Likely effect:
- Higher weight decay: more regularization.
- Lower weight decay: less regularization.

Concrete example:
- If train keeps improving but val/test lag badly, slightly higher decay can help.
- If the model seems over-constrained and not fitting well enough, slightly lower decay can help.

Practical note:
- In your recent runs, lowering weight decay from `1e-4` to `5e-5` did not help. That suggests `1e-4` is a better current baseline for your setup.

---

### `clip_gradients_norm`
What it does:
- Clips gradient norms to improve stability.

When to change it:
- If you see training instability or exploding gradients.

Likely effect:
- Mostly a stability safeguard.

Concrete example:
- `10.0` is a normal, sensible value. I would not touch it unless you have evidence of gradient problems.

---

### `moving_average_decay`
What it does:
- Controls exponential moving average of weights.

When to change it:
- Rarely.

Likely effect:
- Can improve evaluation stability.

Concrete example:
- I would leave this alone unless you are doing a deeper training-system investigation.

---

## H. Mixed precision

### `mixed_precision`
What it does:
- Enables mixed precision training in the EfficientDet config.

When to change it:
- When your environment supports it and you want speed / memory gains.

Likely effect:
- Can reduce memory use and improve speed on supported hardware.
- Can introduce dtype issues if the codebase and TensorFlow/Keras setup are not aligned.

Concrete example:
- On modern NVIDIA GPUs, mixed precision often helps performance and can allow larger batches.
- If enabling it causes compile or dtype errors, that is usually an environment / compatibility problem, not a model-quality issue.

Practical recommendation:
- Treat mixed precision as infrastructure work, not the first AP-improvement lever.

---

### `loss_scale`
What it does:
- Helps avoid numerical underflow during mixed-precision training.

When to change it:
- Only when mixed precision is actually enabled and your stack needs it.

Likely effect:
- Mostly stability under lower-precision arithmetic.

Concrete example:
- If AMP is off, this is irrelevant.

---

## I. Postprocessing / evaluation

### `nms_configs['iou_thresh']`
What it does:
- Controls how aggressively overlapping detections suppress each other.

When to change it:
- During postprocessing sweeps, not during core training ablations.

Likely effect:
- Lower IOU threshold: stronger suppression.
- Higher IOU threshold: more overlapping boxes survive.

Concrete example:
- If multiple boxes around the same dumbbell survive too often, lower it.
- If true detections are being suppressed too aggressively, raise it a bit.

Practical recommendation:
- Fix NMS settings while tuning training hparams. Sweep NMS separately on the best checkpoint.

---

### `nms_configs['score_thresh']`
What it does:
- Minimum confidence score for detections to survive postprocessing.

When to change it:
- During evaluation/postprocessing analysis, not to change what the model learns.

Likely effect:
- Lower score threshold: better recall, more false positives.
- Higher score threshold: cleaner predictions, lower recall.

Concrete example:
- Lowering from `0.6` to `0.4` can raise recall and improve AP on a fixed checkpoint.
- That does not necessarily mean the model itself learned better.

Practical recommendation:
- Freeze this while doing training comparisons.

---

### `nms_configs['method']`, `sigma`, `max_output_size`, `max_nms_inputs`
What they do:
- Control the NMS implementation details and limits.

When to change them:
- Usually later, during deployment or checkpoint polish.

Likely effect:
- Mostly postprocessing behavior.

Concrete example:
- These are rarely the first knobs to turn on a one-class detector.

---

# 4) What matters most for your current dumbbell detector

Based on the behavior you described, the most important knobs for you right now are probably:

1. `num_epochs`
2. `learning_rate`
3. `box_loss_weight`
4. anchors (`aspect_ratios`, `anchor_scale`, `num_scales`)
5. `label_smoothing`
6. fixed NMS for fair comparison

The lower-priority knobs right now are:
- `alpha`
- `gamma` (unless classification balance becomes the clear problem)
- `delta` (unless box-regression penalty shape becomes the clear problem)
- `momentum`
- `clip_gradients_norm`
- `moving_average_decay`

---

# 5) Concrete example tuning playbook

## Situation 1: AP50 is high, AP75 lags
Try in this order:
1. better anchors
2. box loss weight up a bit
3. slightly lower LR for refinement
4. larger image size if memory allows
5. better label curation

## Situation 2: The model still peaks late
Try:
1. extend epochs moderately, for example 200 -> 225
2. do not also change many other knobs at once
3. see where best epoch lands

## Situation 3: False positives are the main problem
Try:
1. check NMS / score threshold first
2. then revisit gamma only if this is clearly a training-side classification issue

## Situation 4: Model confidence feels too soft
Try:
1. lower label smoothing a little
2. keep the change small
3. watch AP and AP75, not just loss

## Situation 5: Loss looks great but AP is not improving
Do not assume the lower loss is better.
For detector tuning, checkpoint selection by:
- validation AP
- validation AP75
is often more useful than selecting by validation loss alone.

---

# 6) Short answers to common questions

## “When do I adjust alpha?”
Late, and only if class balance between positives and negatives is clearly the issue.

## “When do I adjust gamma?”
When classification hardness / false positive / false negative behavior is the actual bottleneck.
Not when the main issue is loose boxes.

## “When do I adjust delta?”
When localization loss behavior needs changing after anchors, LR, and box weighting are already fairly settled.

## “What is usually worth tuning first?”
For your kind of project:
- anchors
- learning rate
- epochs
- box loss weight
- label smoothing
- data curation

---

# 7) Source notes

This guide is based on:
- the EfficientDet default hyperparameter definitions in `hparams_config.py`
- the EfficientDet training/eval logic in `det_model_fn.py`
- TensorFlow mixed precision guidance
- TensorFlow Huber loss definition

