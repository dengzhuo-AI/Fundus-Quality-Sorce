
# FTHNet: Fundus Image Quality Assessment Using Transformer-based Hyper Network on a New Clinical Benchmark

# FTHNET_TEST_DEMO

The test codes for the FTHNet.

## Dependencies

See the requirement for more details.

## sample_FQS

20% of the FQS data randomly selected in sample_FQS.
You can download the samples in releases v1.0.0.

## Usages


Predicting image quality with our model trained on the EyeQS Dataset. (Single GPU test is recommanded for its low occupation.)


```
CUDA_VISIBLE_DEVICES=XXXXX python testdemo.py
```

You will get a quality score ranging from 0-1, and a higher value indicates better image quality.

# The train codes and complete dataset will be released to the public soon!
